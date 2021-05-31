import json
import argparse
import os
import sys
import math
import warnings
import pprint as pp
from tqdm import tqdm
from typing import List, Dict

from datasets import load_dataset, load_metric

def convert(args):
    with open(args.source_file) as f:
        source = json.load(f)
    with open(args.prediction_file) as f:
        predictions = json.load(f)
    
    if args.position_file is not None:
        with open(args.position_file, "r") as f:
            positions = json.load(f)

    outputs = []
    filtered_positions = {}
    for item in source:
        _item = {"id": item["id"], "prediction_text": "", "no_answer_probability":1}
        try:
            _item["prediction_text"] = predictions[item["id"]]
            if len(_item["prediction_text"]) > 0:
                _item["no_answer_probability"] = 0
            outputs.append(_item)

            if args.position_file is not None:
                filtered_positions.update({item["id"]: positions[item["id"]]})
        except KeyError:
            print(item["id"], 'not found!')
    
    with open(args.output_file, 'w') as f:
        json.dump(outputs, f)
    
    if args.position_file is not None:
        with open(args.position_file.replace(".json", "_filtered.json"), 'w') as f:
            json.dump(filtered_positions, f)

def filter_save(source_file, save_name, preds, do_filter=True):
    if not do_filter:
        with open(save_name, "w") as f:
            json.dump(preds, f)
        return None

    with open(source_file) as f:
        source = json.load(f)

    outputs = []
    for item in source:
        target_id = item["id"]
        _item = None
        for i in preds:
            if i["id"] == target_id:
                _item = i
                break
        if _item is not None:
            outputs.append(i)
        else:
            raise ValueError(f"{target_id} not found!")
    with open(save_name, "w") as f:
        json.dump(outputs, f)

def insert_missing_spans(spans):
    add_spans = []
    for i in range(1, len(spans)):
        if spans[i][0] > spans[i-1][1]:
            add_spans.append((spans[i-1][1], spans[i][0]))
    spans.extend(add_spans)
    return sorted(spans, key=lambda tup: tup[0])

def prepare_doc_info_from_doc_datasets(data):
    info = {}
    dummy = {}
    ids = data["doc_id"]
    spans = data["spans"]
    for doc_id, span in zip(ids, spans):
        info_item = []
        id_secs = []
        for s in span:
            id_sec = s["id_sec"]
            if id_sec not in id_secs:
                id_secs.append(id_sec)
                info_item.append(s["text_sec"])
        info[doc_id] = info_item
        dummy[doc_id] = []
    return info, dummy

def prepare_spans_from_doc_datasets(data):
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


def prepare_data_from_datasets(data, sp_spans, sec_spans, do_eval=True):
    proc_data = {}
    if do_eval:
        for doi, title, domain, context, question, answer in zip(data["id"], data["title"], data["domain"], data["context"], data["question"], data["answers"]):
            proc_data[doi] = {"context": context, "domain": domain, "question": question, "sec_spans": sec_spans[title], "sp_spans": sp_spans[title], "gold": answer}
    else:
        for doi, title, domain, context, question in zip(data["id"], data["title"], data["domain"], data["context"], data["question"]):
            proc_data[doi] = {"context": context, "domain": domain, "question": question, "sec_spans": sec_spans[title], "sp_spans": sp_spans[title], "gold": None}
    return proc_data


def get_spans(answer_start, answer_end, spans, threshold=0.5):
    answer_span = []
    for idx, span in enumerate(spans):
        span_length = span[1] - span[0]
        if not (answer_end <= span[0] or answer_start >= span[1]): # whether the span is included by the answer
            if answer_start >= span[0] and answer_end <= span[1]:
                coverage = span_length
            elif answer_start <= span[0] and answer_end <= span[1]:
                coverage = answer_end - span[0]
            elif answer_end >= span[1] and span[1] >= answer_start >= span[0]:
                coverage = span[1] - answer_start
            else:
                coverage = span_length
            ratio = coverage / span_length
            if ratio >= threshold:
                answer_span.append(idx)
    return answer_span


def get_answers(context, raw_answer, sp_spans, sec_spans, threshold=0.5, return_span=False):
    if isinstance(raw_answer, List) and raw_answer[1]-raw_answer[0]==0:
        raw_answer = ""

    if raw_answer == "empty" or raw_answer == "":
        answer_sp = ""
        answer_sec = ""
    else:
        # question answering one by one
        if isinstance(raw_answer, List):
            answer_start, answer_end = raw_answer[0], raw_answer[1]
        else:
            answer_start = context.find(raw_answer, 0)
            answer_end = answer_start + len(raw_answer)
        
        answer_sec_span = get_spans(answer_start, answer_end, sec_spans, threshold=threshold/10)
        answer_sp_span = get_spans(answer_start, answer_end, sp_spans, threshold=threshold)
        answer_sp = context[sp_spans[answer_sp_span[0]][0]:sp_spans[answer_sp_span[-1]][1]] if not return_span else (sp_spans[answer_sp_span[0]][0], sp_spans[answer_sp_span[-1]][1])
        answer_sec = context[sec_spans[answer_sec_span[0]][0]:sec_spans[answer_sec_span[-1]][1]] if not return_span else (sec_spans[answer_sec_span[0]][0], sec_spans[answer_sec_span[-1]][1])

        # assert raw_answer in answer_sp
        # assert raw_answer in answer_sec

    return answer_sp, answer_sec


def eval_metrics(metric, preds, proc_eval_dataset, do_eval=True, threshold=0.5, return_span=False):
    """
    preds [
        {"id": k, "prediction_text": v, "no_answer_probability": 0}
    ]
    refs [
        {"id": k, "answers": answer}
    ]
    """
    raw_preds = []
    sp_preds = [] if not return_span else {}
    sec_preds = [] if not return_span else {}
    refs = []

    for _id, answer in tqdm(preds.items(), total=len(preds)):
        context = proc_eval_dataset[_id]["context"]
        sp_spans = proc_eval_dataset[_id]["sp_spans"]
        sec_spans = proc_eval_dataset[_id]["sec_spans"]
        sp_answer, sec_answer = get_answers(context, answer, sp_spans, sec_spans, threshold=threshold, return_span=return_span)
        
        if return_span:
            if answer[0] == answer[1] == 0:
                sp_preds[_id] = (0,0)
                sec_preds[_id] = (0,0)
            else:
                sp_preds[_id] = sp_answer
                sec_preds[_id] = sec_answer
        else:
            answer = context[answer[0]:answer[1]] if type(answer)==list else answer
            refs.append({"id": _id, "answers": proc_eval_dataset[_id]["gold"]})
            raw_preds.append({"id": _id, "prediction_text": answer, "no_answer_probability": 0})
            sp_preds.append({"id": _id, "prediction_text": sp_answer, "no_answer_probability": 0})
            sec_preds.append({"id": _id, "prediction_text": sec_answer, "no_answer_probability": 0})

    results = {}
    if do_eval:
        results["raw"] = metric.compute(predictions=raw_preds, references=refs)
        results["sp"] = metric.compute(predictions=sp_preds, references=refs)
        results["sec"] = metric.compute(predictions=sec_preds, references=refs)
    return sp_preds, sec_preds, results


def get_split(args):
    if args.split == 'validation':
        datasets = load_dataset("../utils/dialdoc/dialdoc.py", "doc2dial_rc")
        split = args.split
    elif args.split == 'testdev':
        datasets = load_dataset("../utils/dialdoc/dialdoc.py", "doc2dial_rc_testdev")
        split = 'validation'
        if args.do_eval:
            warnings.warn('Cannot do evaluation with this split. Evaluation ignored...', UserWarning)
            args.do_eval = False
    elif args.split == 'test':
        datasets = load_dataset("../utils/dialdoc/dialdoc.py", "doc2dial_rc_test")
        split = 'validation'
        if args.do_eval:
            warnings.warn('Cannot do evaluation with this split. Evaluation ignored...', UserWarning)
            args.do_eval = False
    else:
        raise NotImplementedError
    doc_dataset = load_dataset("../utils/dialdoc/dialdoc.py", name="document_domain" if args.split!="test" else "document_domain_test", split="train")

    with open(args.prediction_file, "r") as f:
        preds = json.load(f)

    split_dataset = datasets[split]
    sp_spans, sec_spans = prepare_spans_from_doc_datasets(doc_dataset)
    proc_eval_dataset = prepare_data_from_datasets(split_dataset, sp_spans, sec_spans, args.do_eval)


    metric = load_metric("squad_v2")
    sp_preds, sec_preds, results = eval_metrics(metric, preds, proc_eval_dataset, args.do_eval, threshold=args.threshold, return_span=args.save_span)
    
    if args.save:
        filter_save(args.source_file, args.prediction_file.replace(".json", "_sp.json"), sp_preds, do_filter=args.do_filter)
    
    if args.save_span:
        with open(args.output_file, "w") as writer:
            writer.write(json.dumps(sp_preds, indent=4) + "\n")

    if args.do_eval:
        pp.pprint(results)


def get_yes_no_answers(question_str, answer, info, gold=None):
    new_answer = ""
    text = question_str.replace("user: ", "")
    if not answer.startswith("Yes") and not answer.startswith("No"):
        if text.startswith('do') or text.startswith('can') or text.startswith('is') or text.startswith('will') or text.startswith('am'):
            # print("question|", question_str)
            # print("answer|", answer)
            # print("gold|", gold)
            for spans in info:
                if answer in spans:
                    if "Yes. "+answer in spans:
                        new_answer = "Yes. "+answer
                        break
                    if "Yes , "+answer in spans:
                        new_answer = "Yes , "+answer
                        break
                    if "No , "+answer in spans:
                        new_answer = "No , "+answer
                        break
                    if "No . "+answer in spans:
                        new_answer = "No . "+answer
                        break
            # if len(new_answer) > 0:
            #     print("new_answer", new_answer)
            #     input()
    new_answer = answer if len(new_answer) == 0 else new_answer
    return new_answer


def yes_no_adding(metric, preds, proc_eval_dataset, do_eval):
    results = {}
    raw_preds = []
    new_preds = []
    refs = []

    for item in tqdm(preds, total=len(preds)):
        if type(item) == str:
            _id = item
            answer = preds[item]
        else:
            _id = item["id"]
            answer = item["prediction_text"]

        question = proc_eval_dataset[_id]["question"]
        domain = proc_eval_dataset[_id]["domain"]
        if domain == "dmv":
            info = proc_eval_dataset[_id]["sp_spans"]
            new_answer = get_yes_no_answers(question.lower(), answer, info, gold=proc_eval_dataset[_id]["gold"]["text"][0] if proc_eval_dataset[_id]["gold"] is not None else None)
        else:
            new_answer = answer

        refs.append({"id": _id, "answers": proc_eval_dataset[_id]["gold"]})
        new_preds.append({"id": _id, "prediction_text": new_answer, "no_answer_probability": 0})

    if do_eval:
        results["raw"] = metric.compute(predictions=preds, references=refs)
        results["yes_no"] = metric.compute(predictions=new_preds, references=refs)
    return new_preds, results
    

def patch(args):
    # get the domain
    if args.split == 'validation':
        datasets = load_dataset("../utils/dialdoc/dialdoc.py", "doc2dial_rc")
        split = args.split
    elif args.split == 'testdev':
        datasets = load_dataset("../utils/dialdoc/dialdoc.py", "doc2dial_rc_testdev")
        split = 'validation'
        if args.do_eval:
            warnings.warn('Cannot do evaluation with this split. Evaluation ignored...', UserWarning)
            args.do_eval = False
    elif args.split == 'test':
        datasets = load_dataset("../utils/dialdoc/dialdoc.py", "doc2dial_rc_test")
        split = 'validation'
        if args.do_eval:
            warnings.warn('Cannot do evaluation with this split. Evaluation ignored...', UserWarning)
            args.do_eval = False
    else:
        raise NotImplementedError
    
    doc_dataset = load_dataset("../utils/dialdoc/dialdoc.py", name="document_domain" if args.split!="test" else "document_domain_test", split="train")
    with open(args.prediction_file, "r") as f:
        preds = json.load(f)
    
    split_dataset = datasets[split]
    info, dummy = prepare_doc_info_from_doc_datasets(doc_dataset)
    proc_eval_dataset = prepare_data_from_datasets(split_dataset, info, dummy, args.do_eval)

    metric = load_metric("squad_v2")
    if args.subtask == "yes_no":
        new_preds, results = yes_no_adding(metric, preds, proc_eval_dataset, args.do_eval)
    else:
        raise NotImplementedError
    
    if args.save:
        with open(args.prediction_file.replace(".json", "_yesno.json"), "w") as f:
            json.dump(new_preds, f)

    if args.do_eval:
        pp.pprint(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task type: [convert]: convert to submission format; [split]: get the full split of the predictions",
    )
    parser.add_argument('--prediction_file', help='path to the prediction file', type=str)
    parser.add_argument('--source_file', default='../data/test_dev/test_subtask1_phase1_ids.json', type=str)

    # convert
    parser.add_argument('--position_file', help='path to the position file', type=str, default=None)
    parser.add_argument('--output_file', help='path to desired output file', type=str)
    
    # split
    parser.add_argument("--do_eval", action="store_true", help="Compute subtask1 metrics with three types of answers",)
    parser.add_argument("--split", default="validation", type=str, help='Data split for validation that is either "validation" "devtest" or "test"',)
    parser.add_argument("--do_filter", action="store_true", help="Filter out additional predictions",)
    parser.add_argument("--save", action="store_true", help="save the sp and sec predictions",)
    parser.add_argument("--save_span", action="store_true", help="save the sp and sec positions",)
    parser.add_argument('--threshold', help='threshold to filter out the sp', type=float, default=0.5)

    # patch
    parser.add_argument("--subtask", type=str, default="yes_no", help="subtask under patching")
    args = parser.parse_args()

    if args.task == "convert":
        convert(args)
    elif args.task == "split":
        get_split(args) 
    elif args.task == "patch":
        patch(args)
    else:
        raise NotImplementedError