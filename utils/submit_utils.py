import json
import argparse

def convert(args):
    with open(args.prediction_json, "r") as  f:
        preds = json.load(f)
    
    gens = []
    if type(preds) == dict:
        for id, text in preds.items():
            gen = {"id": id, "utterance": text}
            gens.append(gen)
    elif type(preds) == list:
        for pred in preds:
            gen = {"id": pred["id"], "utterance": pred["prediction_text"]}
            gens.append(gen)

    with open(args.output_json, "w") as f:
        json.dump(gens, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_json",
        type=str,
        required=True,
        help="Path to predictions",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to outputs",
    )
    args = parser.parse_args()

    convert(args)

if __name__ == "__main__":
    main()