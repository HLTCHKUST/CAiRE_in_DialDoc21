import itertools
import json
import linecache
import math
import os
import pickle
import socket
import random
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

from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer
from transformers.file_utils import cached_property

from .sentence_splitter import add_newline_to_end_of_each_sentence
from .sampler import sortish_sampler_indices, SortishSampler, DistributedSortishSampler
from .data_preproc import load_doc2dial_data, prepare_lines

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dial_data,
        doc_data, 
        max_source_length,
        max_target_length,
        split="train",
        doc_mode="full",
        n_obs=None,
        prefix="",
        history_len=3,
        preds=None,
        **dataset_kwargs
    ):
        super().__init__()
        assert dial_data is not None and doc_data is not None
        ids, src, tgt = prepare_lines(dial_data, doc_data, doc_mode, history_len, split=split, preds=preds)
        # if n_obs is not None:
        #     ids, src, tgt = zip(*random.sample(list(zip(ids, src, tgt)), n_obs))
        self.ids = ids
        self.src = src
        self.tgt = tgt

        self.src_lens = self.get_char_lens(self.src)
        self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        # assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data):
        return [len(x) for x in data]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt) 

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __getitem__(self, index) -> Dict[str, str]:
        return {"tgt_texts": self.tgt[index], "src_texts": self.src[index]}
        # return self.tokenize_function(self.src[index], self.tgt[index])
    
    def pad_or_truncate(self, input_ids, attention_mask, max_length):
        len_to_pad = max_length - len(input_ids) 
        if len_to_pad >= 0:
            input_padding = [self.pad_token_id] * len_to_pad
            input_ids = input_ids + input_padding 

            attention_mask_padding = [0] * len_to_pad
            attention_mask = attention_mask + attention_mask_padding 
        else:
            # truncate
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        return input_ids, attention_mask

    def tokenize_function(self, src, tgt):
        src, tgt = src.strip(), tgt.strip()

        input_ids = self.tokenizer.encode(src)
        attention_mask = [1]*len(input_ids)
            
        # encode tgt
        labels = self.tokenizer.encode(tgt)
        label_mask = [1]*len(labels)

        # pad or truncate the sequence
        input_ids, attention_mask = self.pad_or_truncate(input_ids, attention_mask, self.max_source_length)
        labels, label_mask = self.pad_or_truncate(labels, label_mask, self.max_target_length)
        decoder_input_ids = labels.copy()

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        decoder_input_ids =  torch.LongTensor(decoder_input_ids[:-1])

        label_mask = torch.LongTensor(label_mask[1:])
        labels = torch.LongTensor(labels[1:])
        labels[~label_mask.bool()] = -100
        # print({"input_ids":input_ids, "attention_mask":attention_mask, "decoder_input_ids":decoder_input_ids, "labels": labels})
        # input()
        return {"input_ids":input_ids, "attention_mask":attention_mask, "decoder_input_ids":decoder_input_ids, "labels": labels}


class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.dataset_kwargs = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
        if data_args.src_lang is not None:
            self.dataset_kwargs["src_lang"] = data_args.src_lang
        if data_args.tgt_lang is not None:
            self.dataset_kwargs["tgt_lang"] = data_args.tgt_lang

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            batch = self._encode(batch)
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
            decoder_input_ids = labels.clone()[:, :-1]
            labels = self.ignore_pad_token_for_loss(labels, self.pad_token_id)[:, 1:]

        else:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch])

            labels = trim_batch(labels, self.pad_token_id)
            input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)
            decoder_input_ids = labels.clone()[:, :-1]
            labels = self.ignore_pad_token_for_loss(labels, self.pad_token_id)[:, 1:]

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_input_ids":decoder_input_ids,
        }
        return batch

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding="max_length" if self.tpu_num_cores is not None else "longest",  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        return batch_encoding.data

    def ignore_pad_token_for_loss(self, labels, pad_token_id):
        label_mask = labels.eq(pad_token_id)
        labels[label_mask.bool()] = -100
        return labels
