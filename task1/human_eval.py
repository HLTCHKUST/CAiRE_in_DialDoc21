import argparse
import collections
import json
import os
import re
import string
import sys
import math
import warnings

import numpy as np
import pprint as pp
from tqdm import tqdm
from typing import List, Dict

from datasets import load_dataset, load_metric

def insert_missing_spans(spans):
    add_spans = []
    for i in range(1, len(spans)):
        if spans[i][0] > spans[i-1][1]:
            add_spans.append((spans[i-1][1], spans[i][0]))
    spans.extend(add_spans)
    return sorted(spans, key=lambda tup: tup[0])


def prepare_doc_from_datasets(data):
    spans = data["spans"]
    ids = data["doc_id"]
    sp_spans = {}
    sec_spans = {}
    for doc_id, span in zip(ids, spans):
        if doc_id not in sp_spans:
            sp_span = []
            for split in span:
                # add the span into sp_spans if it's not in the list
                if (split["start_sp"], split["end_sp"]) not in sp_span:
                    sp_span.append((split["start_sp"], split["end_sp"]))
            sp_span = insert_missing_spans(sorted(sp_span, key=lambda tup: tup[0]))
            sp_spans[doc_id] = sp_span
        
        if doc_id not in sec_spans:
            sec_span = []
            for split in span:
                # add the span into sp_spans if it's not in the list
                if (split["start_sec"], split["end_sec"]) not in sec_span:
                    sec_span.append((split["start_sec"], split["end_sec"]))
            sec_span = insert_missing_spans(sorted(sec_span, key=lambda tup: tup[0]))
            sec_spans[doc_id] = sec_span
    return sp_spans, sec_spans


def prepare_data_from_datasets(data, sp_spans, sec_spans, split):
    proc_data = {}
    if split != 'test':
        for doi, title, context, question, answer in zip(data["id"], data["title"], data["context"], data["question"], data["answers"]):
            proc_data[doi] = {"context": context, "sec_spans": sec_spans[title], "sp_spans": sp_spans[title], "question": question, "gold": answer}
    else:
        for doi, title, context, question in zip(data["id"], data["title"], data["context"], data["question"]):
            proc_data[doi] = {"context": context, "sec_spans": sec_spans[title], "sp_spans": sp_spans[title], "question": question, "gold": ''}
    return proc_data


# normalize answer
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def analysis(prediction_file, split="test", compare=None):
    metric = load_metric("squad_v2")

    datasets = load_dataset("../utils/dialdoc/dialdoc.py", "doc2dial_rc_test")
    doc_dataset = load_dataset("../utils/dialdoc/dialdoc.py", name="document_domain_test", split="train")

    with open(prediction_file, "r") as f:
        preds = json.load(f)
    
    if compare is not None:
        with open(compare, "r") as f:
            comp = json.load(f)
        if type(comp) == list:
            comp_dict = {}
            for item in comp:
                comp_dict[item["id"]] = item["prediction_text"]
        else:
            comp_dict = comp

    eval_dataset = datasets['validation']
    sp_spans, sec_spans = prepare_doc_from_datasets(doc_dataset)
    proc_eval_dataset = prepare_data_from_datasets(eval_dataset, sp_spans, sec_spans, split=split)
    
    diff = []
    for x in tqdm(preds, total=len(preds)):
        if type(preds) == list:
            _id = x["id"]
            answer = x["prediction_text"]
        else:
            _id = x
            answer = preds[x]
        context = proc_eval_dataset[_id]["context"]
        question = proc_eval_dataset[_id]["question"]
        sp_spans = proc_eval_dataset[_id]["sp_spans"]
        sec_spans = proc_eval_dataset[_id]["sec_spans"]
        if split != "test":
            gold_answer = proc_eval_dataset[_id]["gold"]["text"][0]
        else:
            gold_answer = ''

        exact = compute_exact(answer, gold_answer)
        f1 = compute_f1(answer, gold_answer)
        if exact != 1:
            item = {}
            item["context"] = context
            item["question"] = question
            item["gold"] = [gold_answer]
            item["pred"] = [answer]
            item["compare"] = comp_dict[_id]
            item["exact"] = exact
            item["f1"] = f1
            diff.append(item)
    
    print(f"There are {len(diff)} / {len(preds)} predictions are not fully correct.")
    for item in diff:
        for k, v in item.items():
            print(k, v)
            if k == "context":
                print("-"*80)
        print("\n\n")
        input()

analysis("save/deepset-roberta-base-squad2/test/positions_sp.json", compare="../results/sp_test2_ensemble_preds.json")