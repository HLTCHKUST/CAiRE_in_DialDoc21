import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt 
import numpy as np

from datasets import load_dataset

DOC_DOMAIN_SPLIT = "train"
YOUR_DATASETS_SOURCE_DIR = ""  # the root folder of your local `datasets` source code.

def text2line(text):
    return text.replace("\n", " ").replace("\r", " ").strip()


def save_dial_domain():
    dial_dataset = load_dataset("../utils/dialdoc/dialdoc.py", name=f"dialogue_domain_test", split=DOC_DOMAIN_SPLIT)
    d = {}
    for ex in tqdm(dial_dataset, total=len(dial_dataset), ncols=100):
        doc_id = ex["doc_id"]
        domain = ex["domain"]
        dial_context = []

        dial_id = ex["dial_id"]
        dial_context = []
        for i, turn in enumerate(ex["turns"]):
            utterance = text2line(turn["utterance"])

        turn_id = turn["turn_id"]
        key = f"{dial_id}_{turn_id}"
        v = {"domain":domain, "type":"id" if domain != "cdccov19" else "ood"}
        d[key] = v
    with open("cache/test_domain.json", "w") as f:
        json.dump(d, f)
    

def load_dial_data(split):
    if "test" in split:
        dial_dataset = load_dataset("../utils/dialdoc/dialdoc.py", name=f"dialogue_domain_{split}", split=DOC_DOMAIN_SPLIT)
    else:
        dial_dataset = load_dataset("../utils/dialdoc/dialdoc.py", name="dialogue_domain", split=split)

    lines = []
    for ex in tqdm(dial_dataset, total=len(dial_dataset), ncols=100):
        doc_id = ex["doc_id"]
        dial_context = []

        dial_id = ex["dial_id"]
        dial_context = []
        for i, turn in enumerate(ex["turns"]):
            utterance = text2line(turn["utterance"])

            if "test" not in split:
                if not turn["references"]:  # this task only uses instances and evalutes on the grounded turns.
                    continue

                if turn["role"] == "agent" and len(dial_context)>0:
                    turn_id = turn["turn_id"]-1
                    src = "[SEP]".join(dial_context)
                    tgt = utterance

                    doc_info = []
                    for ref in turn["references"]:
                        doc_info.append(ref["sp_id"])
                    doc_info.sort(key=lambda x:int(x))
                    doc_info = "[SEP]".join(doc_info)

                    lines.append("|||".join([f"{dial_id}_{turn_id}", src, tgt, doc_id, doc_info]))
            dial_context.append(utterance)
        
        if "test" in split:
            turn_id = turn["turn_id"]
            src = "[SEP]".join(dial_context)
            tgt = "[NULL]"
            doc_info = "[NULL]"
            lines.append("|||".join([f"{dial_id}_{turn_id}", src, tgt, doc_id, doc_info]))

    return lines

def load_doc_data(test_stage):
    doc_dataset = load_dataset("../utils/dialdoc/dialdoc.py", name="document_domain" if test_stage!="test" else "document_domain_test", split=DOC_DOMAIN_SPLIT)

    d_doc = defaultdict(dict)
    for ex in doc_dataset:
        d_doc[ex["doc_id"]]["doc_text"] = ex["doc_text"]
        for d_span in ex["spans"]:
            d_doc[ex["doc_id"]][d_span["id_sp"]] = d_span
    return d_doc

def load_doc2dial_data(do_train, do_eval, do_predict, test_stage="testdev"):
    train_dial_data = load_dial_data("train") if do_train else None
    valid_dial_data = load_dial_data("validation") if do_eval else None
    test_dial_data = load_dial_data(test_stage) if do_predict else None

    dial_data = {
        "train": train_dial_data,
        "validation": valid_dial_data,
        "test": test_dial_data,
    }
    doc_data = load_doc_data(test_stage)
    return dial_data, doc_data

def prepare_input_items(line, doc, grounding, history_len, preds=None, include_doc=True):
    items = line.split("|||")

    assert len(items) == 5
    _id, contexts, response, doc_id, gold_sps = items[0], items[1], items[2], items[3], items[4]

    contexts = contexts.split("[SEP]")[-history_len:]
    gold_sps = gold_sps.split("[SEP]")
    
    if preds is not None:
        if _id not in preds:
            return None, None, None, None
        doc_input = preds[_id]
    elif "[NULL]" in gold_sps or not include_doc:
        doc_input = None
    else:
        doc_text = []
        for sp_id in gold_sps:
            doc_data = doc[doc_id][sp_id]
            if grounding == "sp":
                text = doc_data["text_sp"]
            elif grounding == "sec":
                text = doc_data["text_sec"]
            else:
                raise NotImplementedError

            if text not in doc_text:
                doc_text.append(text)
        doc_input = "".join(doc_text)
    return _id, contexts, response, doc_input


def prepare_lines(dial, doc, grounding, history_len, model_type="seq2seq", tokenizer=None, split="train", preds=None):
    ids, src, tgt = [], [], []
    slens = []
    tlens = []
    for line in dial:
        src_line = []
        _id, contexts, response, doc_input = prepare_input_items(line, doc, grounding, history_len, preds=preds)
        if _id is None:
            continue

        if doc_input is not None:
            doc_input = doc_input.strip() + "<eos_k>"
            src_line.append(doc_input)

        for idx, uttr in enumerate(contexts):
            if idx % 2 == 0:
                appx = "<eos_u>"
            else:
                appx = "<eos_r>"
            uttr = uttr.strip() + appx
            src_line.append(uttr)

        ids.append(_id)
        if tokenizer is not None:
            slens.append(len(tokenizer.encode("".join(src_line))))
            tlens.append(len(tokenizer.encode(response)))

        src.append("".join(src_line) if model_type == "seq2seq" else src_line) 
        tgt.append(response if model_type == "seq2seq" else "<bos>"+response)
    
    if tokenizer is not None:
        print("slen", max(slens), min(slens), np.mean(slens))
        print("tlen", max(tlens), min(tlens), np.mean(tlens))
        plt.figure()
        plt.hist(slens)
        plt.savefig(f'{split}-slen.png')

        plt.figure()
        plt.hist(tlens)
        plt.savefig(f'{split}-tlen.png')
    
    if not os.path.exists(f"cache/{split}_ids.txt"):
        with open(f"cache/{split}_ids.txt", "w") as f:
            for _id in ids:
                f.write(_id+"\n")
    return ids, src, tgt

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    # save_dial_domain()
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--hist', type=int, help='an integer for the accumulator')
    parser.add_argument('--split', type=str, help='an integer for the accumulator')
    parser.add_argument('-dm', '--doc_mode', type=str, help='an integer for the accumulator')
    parser.add_argument("-st", '--stage', type=str, default="testdev")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    dial_data, doc_data = load_doc2dial_data(True, True, True, test_stage=args.stage)
    ids, src, tgt = prepare_lines(dial_data[args.split], doc_data, args.doc_mode, args.hist, model_type="seq2seq", split=(args.split if "test" not in args.split else args.stage))
    # with open("data/val_ids.txt", "w") as f:
    #     for _id in ids:
    #         f.write(_id+"\n")
    # with open("save/test/src-3.txt", "w") as f:
    #     for src_line in src:
    #         f.write(src_line+"\n")
    # with open("save/test/tgt.txt", "w") as f:
    #     for tgt_line in tgt:
    #         f.write(tgt_line+"\n")

    # data = []
    # for _id, src_line, tgt_line in zip(ids, src, tgt):
    #     data.append({"id": _id, "src": src_line, "tgt": tgt_line})
    # with open(f"data/doc-{args.doc_mode}_hist-{args.hist}_{args.split if 'test' not in args.split else args.stage}.json", "w") as f:
    #     json.dump({"version":"v1.0.1", "data":data}, f)