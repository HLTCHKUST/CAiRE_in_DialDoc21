import itertools
import json
import linecache
import math
import os
import pickle
import socket
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import git
import numpy as np
import torch
import torch.distributed as dist
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler

from .sentence_splitter import add_newline_to_end_of_each_sentence
from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer
from transformers.file_utils import cached_property


try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
