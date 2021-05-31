import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--gen_preds", type=str, help="")
parser.add_argument("--raw_preds", type=str, help="")
parser.add_argument("--domain_file", type=str, default="cache/test_domain.json") 
parser.add_argument("--output_file", type=str, help="")

args = parser.parse_args()

with open(args.gen_preds, "r") as f:
    gens = json.load(f)

with open(args.raw_preds, "r") as f:
    raws = json.load(f)

if args.domain_file is not None:
    with open(args.domain_file, "r") as f:
        domain = json.load(f)
else:
    domain = None

mix_gens = []
for gen, raw in zip(gens, raws):
    assert gen["id"] == raw["id"]
    _id = gen["id"]
    d = domain[_id]["type"] if domain is not None else "id"

    if d == "id":
        new = gen
    else:
        ratio = len(gen["utterance"].split()) / len(raw["prediction_text"].split())
        if ratio < 0.4 : # and (len(raw["prediction_text"].split()) > 7 and len(gen["utterance"].split()) > 7 )
            new = {"id":raw["id"], "utterance": raw["prediction_text"]}
        else:
            new = gen
    mix_gens.append(new)

assert len(mix_gens) == len(gens)

with open(args.output_file, "w") as f:
    json.dump(mix_gens, f)
