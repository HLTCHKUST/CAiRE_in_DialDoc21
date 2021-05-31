import json
import os
# subtask 2
from sacrebleu import corpus_bleu
# subtask 1
from datasets import load_metric

def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)

def compute_metrics(preds, refs):
    """
    preds [
        {"id": k, "prediction_text": v, "no_answer_probability": 0}
    ]
    refs [
        {"id": k, "answers": answer}
    ]
    """
    metric = load_metric("squad_v2")
    return metric.compute(predictions=preds, references=refs)

if __name__ == '__main__':
    with open("log/valid/test_seen-decoded-iter-0.txt", "r") as f:
        lines = f.readlines()
    
    hyps = []
    refs = []
    for line in lines:
        hr_pair = line.strip().split("|||")
        assert len(hr_pair) == 2
        hyps.append(hr_pair[0].strip())
        refs.append(hr_pair[1].strip())

    sacrebleu_score = calculate_bleu(hyps, refs)
    print("sacrebleu", sacrebleu_score)
