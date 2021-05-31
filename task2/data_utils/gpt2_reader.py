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

from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer
from transformers.file_utils import cached_property

from .sentence_splitter import add_newline_to_end_of_each_sentence
from .sampler import sortish_sampler_indices, SortishSampler, DistributedSortishSampler
from .data_preproc import load_doc2dial_data

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
    token_type_ids=None,
    labels=None
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask], token_type_ids[:, keep_column_mask], labels[:, keep_column_mask])

class DecoderOnlyDataset(Dataset):
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
        **dataset_kwargs
    ):
        assert dial_data is not None and doc_data is not None
        ids, src, tgt = prepare_lines(dial_data, doc_data, doc_mode, history_len, split=split)
        self.ids = ids[:n_obs] if n_bos is not None else ids
        self.src = src[:n_obs] if n_bos is not None else src
        self.tgt = tgt[:n_obs] if n_bos is not None else tgt

        self.max_source_length = max_source_length
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        self.pad_token_id = self.tokenizer.pad_token_id
        self.sep_token = tokenizer.eos_token
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.tokenize_function(self.src[index], self.tgt[index])
    
    def encoding(self, idx, uttr, input_ids, attention_mask, token_type_ids, label_mask, tgt=False):
        FLAG = idx % 2 # even 0 odd 1
        if tgt:
            uttr = uttr+self.sep_token

        output = self.tokenizer.encode(uttr, add_special_tokens=False)
        input_ids += output
        attention_mask += [1] * len(output)
        token_type_ids += [FLAG] * len(output)
        if tgt:
            label_mask += [0] + [1] * (len(output)-1)
        else:
            label_mask += [0] * len(output)
        return input_ids, attention_mask, token_type_ids, label_mask
    
    def pad(self, input_ids, attention_mask, token_type_ids, label_mask, len_to_pad):
        input_padding = [self.pad_token_id] * len_to_pad
        attention_mask_padding = [0] * len_to_pad
        token_type_padding = [0] * len_to_pad
        label_mask_padding = [0] * len_to_pad

        input_ids = input_ids + input_padding 
        attention_mask = attention_mask + attention_mask_padding 
        token_type_ids = token_type_ids + token_type_padding 
        label_mask = label_mask + label_mask_padding 

        return input_ids, attention_mask, token_type_ids, label_mask

    def tokenize_function(self, src, tgt):
        tgt =  tgt.strip()

        input_ids = []
        attention_mask = []
        token_type_ids = []
        label_mask = []
        # encode src
        pre_add = self.args.history_len-len(src) 
        assert pre_add >= 0
        for idx, uttr in enumerate(src):
            idx += pre_add
            input_ids, attention_mask, token_type_ids, label_mask = self.encoding(idx, uttr, input_ids, attention_mask, token_type_ids, label_mask, tgt=False)
            
        # encode tgt
        input_ids, attention_mask, token_type_ids, label_mask = self.encoding(idx+1, tgt, input_ids, attention_mask, token_type_ids, label_mask, tgt=True)

        # pad or truncate the sequence
        seq_len = len(input_ids) 
        len_to_pad = self.args.max_seq_length - seq_len
        if len_to_pad >= 0:
            input_ids, attention_mask, token_type_ids, label_mask = self.pad(input_ids, attention_mask, token_type_ids, label_mask, len_to_pad)
        else:
            # truncate
            input_ids = input_ids[:self.args.max_seq_length]
            attention_mask = attention_mask[:self.args.max_seq_length]
            token_type_ids = token_type_ids[:self.args.max_seq_length]
            label_mask = label_mask[:self.args.max_seq_length]

        labels = input_ids.copy()

        input_seq_len = self.args.max_seq_length - 1
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)
        label_mask = torch.LongTensor(label_mask)
        labels = torch.LongTensor(labels)
        labels[~label_mask.bool()] = -100

        return {"input_ids":input_ids, "attention_mask":attention_mask, "token_type_ids": token_type_ids, "labels": labels}


class GPT2DataCollator:
    def __init__(self, tokenizer, data_args, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
    
    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        token_type_ids = torch.stack([x["token_type_ids"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])

        labels = trim_batch(labels, self.pad_token_id)
        input_ids, attention_mask, token_type_ids, labels = trim_batch(input_ids, 
                                                                       self.pad_token_id, 
                                                                       attention_mask=attention_mask,
                                                                       token_type_ids=token_type_ids,
                                                                       labels=labels
                                                                       )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        return batch