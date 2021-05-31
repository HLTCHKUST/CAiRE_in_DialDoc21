import logging
import os
import sys
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
from tabulate import tabulate

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    default_data_collator
)
from data_utils import (
    load_doc2dial_data,
    prepare_lines,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_preds", type=str, default=None, help="")
    parser.add_argument("--raw_preds", type=str, default=None, help="")
    parser.add_argument("--split", type=str, default=None, help="")
    parser.add_argument("--stage", type=str, default=None, help="testdev/test")

    args = parser.parse_args()

    # load raw preds and generative preds
    if ".txt" in args.gen_preds:
        with open(args.gen_preds, "r") as f:
            gens = f.readlines()
        with open(args.gen_preds.replace("generation", "gold"), "r") as f:
            golds = f.readlines()
    else:
        with open(args.gen_preds, "r") as f:
            data = json.load(f)
            gens = {}
            for item in data:
                gens[item["id"]] = item["utterance"]
    
    with open(args.raw_preds, "r") as f:
        data = json.load(f)
        raws = {}
        for item in data:
            key = "utterance" if "utterance" in item else "prediction_text"
            raws[item["id"]] = item[key]

    
    dial_data, doc_data = load_doc2dial_data(False, True if args.split=="validation" else False, True if "test" in args.split else False, test_stage=args.stage)

    ids, src, tgt = prepare_lines(dial_data[args.split if args.split!="testdev" else "test"], doc_data, "sp", 3, split=args.split, preds=raws)

    tables = []
    for idx, (_id, s, r) in enumerate(zip(ids, src, tgt)):

        if type(gens) == dict:
            gen = gens[_id]
        else:
            assert golds[idx].strip() == r
            gen = gens[idx].strip()
        raw = raws[_id]
        if raw != gen:
            table = [["id", _id], ["src", s], ["tgt", r], ["generative", gen], ["task1", raw]]

            print(tabulate(table))
            input()