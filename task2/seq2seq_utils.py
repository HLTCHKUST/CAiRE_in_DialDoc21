import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict

from datasets import load_dataset

DOC_DOMAIN_SPLIT = "train"
YOUR_DATASETS_SOURCE_DIR = ""  # the root folder of your local `datasets` source code.

def convert(args):
    ids_path = os.path.join(args.data_dir, f"{args.split}_{args.doc_mode}.ids")
    with open(ids_path, "r") as f:
        ids = f.readlines()

    with open(args.preds_file, "r") as f:
        preds = f.readlines()

    samples = []
    for _id, pred in zip(ids, preds):
        sample = {"id": _id.strip(), "utterance": pred.strip()}
        samples.append(sample)
    print(len(samples))
    sample_path = args.preds_file.replace(".txt", ".json")
    with open(sample_path, "w") as f:
        json.dump(samples, f)


def text2line(text):
    return text.replace("\n", " ").replace("\r", " ").strip()


def btag(tag, text):  # tag the content
    return "<{}>\t{}".format(tag, text2line(text))

def load_doc2dial_seq2seq(args):
    doc_dataset = load_dataset(
        "../utils/dialdoc/dialdoc.py",
        name="document_domain",
        split=DOC_DOMAIN_SPLIT
    )
    dial_dataset = load_dataset(
        "../utils/dialdoc/dialdoc.py",  # path to your datasets source code
        name="dialogue_domain",
        split=args.split,
    )

    if args.doc_mode == "prediction":
        # load the predictions of the output of subtask1
        with open(args.preds_file, "r") as f:
            preds = json.load(f)

    d_doc = defaultdict(dict)
    for ex in doc_dataset:
        d_doc[ex["doc_id"]]["doc_text"] = ex["doc_text"]
        for d_span in ex["spans"]:
            d_doc[ex["doc_id"]][d_span["id_sp"]] = d_span
    source = []
    target = []
    for ex in dial_dataset:
        doc_id = ex["doc_id"]
        d_doc_spans = d_doc[doc_id]
        dial_context = []
        contexts = None
        dial_id = ex["dial_id"]
        for i, turn in enumerate(ex["turns"]):
            if not turn[
                "references"
            ]:  # this task only uses instances and evalutes on the grounded turns.
                continue
            utterance = text2line(turn["utterance"])
            utterance_context = btag(turn["role"], utterance)
            if turn["role"] in args.role:  # if current turn is to predict
                contexts = [
                    btag("last_turn", dial_context[-1].split("\t", 1)[-1])
                ]  # add previous utterance as tagged query context
                contexts.extend(
                    dial_context
                )  # add dialog history as tagged dialogue context
                if args.doc_mode == "full":
                    # add entire document as tagged document context
                    contexts += [
                        btag("title", doc_id),
                        btag("doc_context", d_doc[doc_id]["doc_text"]),
                    ]
                elif args.doc_mode == "grounding":
                    reference_content = ""  # the grounding span content
                    for ref in turn["references"]:
                        sp_id = ref["sp_id"]
                        reference_content += "\t" + d_doc_spans[sp_id]["text_sp"]
                    reference_context = btag("grounding", reference_content)

                    title = btag("title", doc_id)
                    contexts.append(title)
                    contexts.append(reference_context)
                elif args.doc_mode == "grounding_sec":
                    reference_content = ""  # the grounding span content
                    d_sec = {}
                    for ref in turn["references"]:
                        sp_id = ref["sp_id"]
                        sec_id = d_doc_spans[sp_id]["id_sec"]
                        # rename sec_id for sorting the text sections in order.
                        if sec_id.startswith("t"):
                            sec_id = sec_id.split("_", 1)[-1] + "_0"
                        else:
                            sec_id = sec_id + "_1"
                        sec_content = d_doc_spans[sp_id]["text_sec"]
                        d_sec[sec_id] = sec_content
                        reference_content += "\t" + d_doc_spans[sp_id]["text_sp"]
                    sec_contents = []
                    for k, v in sorted(d_sec.items()):
                        sec_contents.append(v)
                        contexts += [
                            btag("title", doc_id),
                            btag(
                                "doc_context", "\t".join(sec_contents)
                            ),  # use a combine of related sections as document context.
                        ]
                    reference_context = btag("grounding", reference_content)
                    contexts.append(reference_context)
                elif args.doc_mode == "null":
                    pass
                elif args.doc_mode == "prediction":
                    sample_id = f"{dial_id}_{turn['turn_id']-1}"
                    if sample_id not in preds:
                        print(sample_id)
                    else:
                        reference_content = preds[sample_id]
                        reference_context = btag("grounding", reference_content)
                        title = btag("title", doc_id)
                        contexts.append(title)
                        contexts.append(reference_context)
                else:
                    raise NotImplementedError
                source.append("\t".join(contexts))
                target.append(utterance)
            dial_context.append(utterance_context)
    assert len(source) == len(
        target
    ), "Need to ensure that source and target are same sized."
    if args.split == "validation":
        args.split = "val"
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    with open(
        os.path.join(args.data_dir, "{}_{}.source".format(args.split, args.doc_mode)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(source)+"\n")
        fp.close()
    with open(
        os.path.join(args.data_dir, "{}_{}.target".format(args.split, args.doc_mode)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(target)+"\n")
        fp.close()



def load_doc2dial_seq2seq_test(args):
    assert "test" in args.split

    # extract the list of ids that requirs for the test set
    target_ids = []
    with open(args.sample_file, "r") as f:
        data = json.load(f)
    for item in data:
        target_ids.append(item["id"])

    # load the predictions of the output of subtask1
    with open(args.preds_file, "r") as f:
        preds = json.load(f)
    matched_preds = {}
    for item in preds:
        matched_preds[item["id"]] = item["prediction_text"].strip()
    
    # load the section predictions
    if "sec" in args.doc_mode:
        assert args.sec_file is not None
        with open(args.sec_file, "r") as f:
            secs = json.load(f)
        matched_secs = {}
        for item in secs:
            matched_secs[item["id"]] = item["prediction_text"].strip()

    doc_dataset = load_dataset(
        "../utils/dialdoc/dialdoc.py",
        name="document_domain",
        split=DOC_DOMAIN_SPLIT
    )
    dial_dataset = load_dataset(
        "../utils/dialdoc/dialdoc.py",  # path to your datasets source code
        name=f"dialogue_domain_{args.split}",
        split=DOC_DOMAIN_SPLIT
    )

    source = []
    target = []
    ids = []
    for ex in dial_dataset:
        dial_context = []
        contexts = None
        dial_id = ex["dial_id"]
        for i, turn in enumerate(ex["turns"]):
            utterance = text2line(turn["utterance"])
            utterance_context = btag(turn["role"], utterance)
            dial_context.append(utterance_context)

        contexts = [
            btag("last_turn", dial_context[-1].split("\t", 1)[-1])
        ]  # add previous utterance as tagged query context
        contexts.extend(
            dial_context
        )  # add dialog history as tagged dialogue context

        sample_id = f"{dial_id}_{turn['turn_id']}"
        assert sample_id in target_ids
        # append prediction behind the dialogue context
        title = btag("title", ex["doc_id"])
        contexts.append(title)

        if "sec" in args.doc_mode:
            sec_context = btag("doc_context", matched_secs[sample_id])
            contexts.append(sec_context)

        reference_context = btag("grounding", matched_preds[sample_id])
        contexts.append(reference_context)

        source.append("\t".join(contexts))
        ids.append(sample_id)
    
    assert len(source) == len(ids) == len(target_ids), "Need to ensure that source and target are same sized."
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    with open(
        os.path.join(args.data_dir, "{}_{}.source".format(args.split, args.doc_mode)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(source)+"\n")
        fp.close()
    with open(
        os.path.join(args.data_dir, "{}_{}.ids".format(args.split, args.doc_mode)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(ids)+"\n")
        fp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="input", help="Build inputs to the model or build the submussion sample file")
    
    parser.add_argument("--split", type=str, required=True, help="Data split is 'train', 'validation' or 'test[dev]'")
    parser.add_argument("--role", type=str, default="agent", help="which role's utterance for generation")
    parser.add_argument("--doc_mode", type=str, default="full", help="whether use entire document")
    parser.add_argument("--data_dir", type=str, required=True, help="path to output the data files")
    
    parser.add_argument("--sample_file", type=str, default="../data/dialdoc21-sharedtask-phase1/test_subtask2_phase1_ids.json", help="path to the sample file")
    parser.add_argument("--preds_file", type=str, default="../save/task1/roberta-large/test_dev/sp_predictions.json", help="path to the sample file")
    parser.add_argument("--sec_file", type=str, default="../save/task1/roberta-large/test_dev/sec_predictions.json", help="path to the sample file")

    args = parser.parse_args()
    print(args)
    if args.task == "input":
        load_dial_data(args)
    elif args.task == "output":
        convert(ids_path, preds_path, sample_path)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
