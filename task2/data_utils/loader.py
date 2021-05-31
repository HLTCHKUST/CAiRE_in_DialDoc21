import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.trainer_utils import EvaluationStrategy, is_main_process

from .seq2seq_reader import Seq2SeqDataCollator, Seq2SeqDataset
from .gpt2_reader import DecoderOnlyDataset
from .data_preproc import load_doc2dial_data

def load_datasets(data_args, model_args, training_args, extra_args, tokenizer, prefix=None):
    do_train = training_args.do_train
    do_eval = training_args.do_eval or extra_args.predict_eval
    do_predict = training_args.do_predict and not extra_args.predict_eval

    dial_data, doc_data = load_doc2dial_data(do_train, do_eval, do_predict, test_stage=data_args.stage)
    if do_predict or extra_args.predict_with_preds:
        assert extra_args.preds_path is not None
        with open(extra_args.preds_path, "r") as f:
            data = json.load(f)
        
        if type(data) == list:
            preds = {}
            for item in data:
                preds[item["id"]] = item["prediction_text"]
        else:
            preds = data
    else:
        preds = None

    dataset_class = Seq2SeqDataset if "gpt" not in model_args.model_name_or_path else DecoderOnlyDataset

    train_dataset = (
        dataset_class(
            tokenizer,
            dial_data["train"],
            doc_data,
            split="train",
            doc_mode=data_args.doc_mode,
            n_obs=data_args.n_train,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=prefix or "",
            history_len=data_args.history_len,
        )
        if do_train
        else None
    )
    eval_dataset = (
        dataset_class(
            tokenizer,
            dial_data["validation"],
            doc_data,
            split="val",
            doc_mode=data_args.doc_mode,
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=prefix or "",
            history_len=data_args.history_len,
            preds=preds,
        )
        if do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )
    test_dataset = (
        dataset_class(
            tokenizer,
            dial_data["validation"] if extra_args.predict_eval else dial_data["test"],
            doc_data,
            split=data_args.stage,
            doc_mode=data_args.doc_mode,
            n_obs=data_args.n_test,
            max_target_length=data_args.test_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=prefix or "",
            history_len=data_args.history_len,
            preds=preds, 
        )
        if do_predict
        else None
    )
    return train_dataset, eval_dataset, test_dataset