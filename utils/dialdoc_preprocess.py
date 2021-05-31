import json
import os
import argparse
from collections import defaultdict

from datasets import load_dataset

DOC_DOMAIN_SPLIT = "train"
YOUR_DATASETS_SOURCE_DIR = ""  # the root folder of your local `datasets` source code.

END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")"]  # acceptable ways to end a sentence


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    return line + " ."

def text2line(text):
    return text.replace("\n", "").replace("\r", "").strip()

def _parse_knowledge(kns:list, correct_first:bool, ground_id:list):
    # we wish the knowledge sentences to keep their original order
    # # we want the correct knowledge to always be in index 0
    # # there could be multiple references
    ground_id.sort()
    if correct_first:
        anchor = 0
        for i in ground_id:
            i = int(i) - 1
            kns[anchor], kns[i] = kns[i], kns[anchor]
            anchor += 1
    return kns

def load_dataset_from_file(filename):
    # read the data file
    with open(filename, "r") as f:
        data = json.load(f)

    dial_dataset = []
    for _, domain_data in data.items():
        for _, v in domain_data.items():
            dial_dataset.extend(v)
    return dial_dataset

def load_doc2dial_seq2seq(args, correct_first=False, keep_last_n=2, grounding=False):
    doc_dataset = load_dataset("../dialdoc", name="document_domain", split=DOC_DOMAIN_SPLIT) # path to your datasets source code
    if args.split == "testdev" or args.split == "test":
        dial_dataset = load_dataset_from_file(args.in_file)
    else:
        dial_dataset = load_dataset( "../dialdoc", name="dialogue_domain", split=args.split, ignore_verifications=True)

    d_doc = {}
    d_doc_span = {}
    for ex in doc_dataset:
        d_doc[ex["doc_id"]] = []
        d_doc_span[ex["doc_id"]] = {}
        for d_span in ex["spans"]:
            # id: d_span["id_sp"]
            d_doc_span[ex["doc_id"]][d_span["id_sp"]] = d_span["text_sp"].replace("\n", "")
            d_doc[ex["doc_id"]].append(d_span["text_sp"].replace("\n", ""))

    for ex in dial_dataset:
        history_strings = []
        users = []
        for i, turn in enumerate(ex["turns"]):
            if not turn.get("references", None):  # this task only uses instances and evalutes on the grounded turns.
                if "test" not in args.split:
                    continue
                else: # current we are in the test set and reference is missing by default
                    turn["references"] = [{"sp_id":0}]
            
            ground_id = []
            for ref in turn["references"]:
                ground_id.append(ref["sp_id"])

            utterance = fix_missing_period(text2line(turn["utterance"]))
            if turn["role"] == "agent":
                users.append(1)
            elif turn["role"] == "user":
                users.append(0)
            else:
                raise ValueError("Invalid role!")
            
            history_strings.append(utterance)

            if turn["role"] == "agent" and "test" not in args.split:
                knowledge = _parse_knowledge(d_doc[ex["doc_id"]], correct_first, ground_id)
                label = utterance
                yield (history_strings[-(keep_last_n+1):-1], users[-(keep_last_n+1):-1], label, knowledge)
        
        if "test" in args.split:
            knowledge = _parse_knowledge(d_doc[ex["doc_id"]], correct_first, ground_id)
            _id = ex["dial_id"] + "_" + str(turn["turn_id"])
            yield (history_strings[-keep_last_n:], users[-keep_last_n:], _id, knowledge)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-training for Knowledge-Grounded Conversation'
    )
    parser.add_argument('--in_file', type=str, default='')
    parser.add_argument('--out_file', type=str, default='')
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Data split is 'train', 'validation' or 'test'",
    )
    args = parser.parse_args()
    
    with open(args.out_file, 'w', encoding='utf-8') as f:
        for history, user, response, knowledge in load_doc2dial_seq2seq(args, correct_first=True, keep_last_n=2):
            f.write(
                json.dumps({
                    'history': history,
                    'user': user,
                    'response': response,
                    'knowledge': knowledge
                }) + '\n'
            )