# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import json
import linecache
import math
import os
import pickle
import socket
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import git
import numpy as np
import torch
import torch.distributed as dist
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler

from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer
from transformers.file_utils import cached_property

from .sentence_splitter import add_newline_to_end_of_each_sentence

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


ATTR_TO_SPECIAL_TOKEN = {'pad_token': '<pad>', 
                         'additional_special_tokens': ['<eos_u>', '<eos_r>', '<eos_k>']}

def add_special_tokens_(tokenizer, model, decoder_only=False):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    if decoder_only:
        ATTR_TO_SPECIAL_TOKEN.update({"bos_token":"<bos>"})
    orig_num_tokens = len(tokenizer) #len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}


def build_compute_metrics_fn(task_name: str, tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)
    
    def recover_pad_token(tokens: np.ndarray) -> np.ndarray:
        tokens[tokens == -100] = tokenizer.pad_token_id
        return tokens

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        labels = lmap(recover_pad_token, pred.label_ids)
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        # with open("save/test/preds.txt", "w") as f:
        #     for line in pred_str:
        #         f.write(line+"\n")
        # with open("save/test/golds.txt", "w") as f:
        #     for line in label_str:
        #         f.write(line+"\n")
        # print("-"*80)
        # for p, g in zip(pred_str, label_str):
        #     print("gold", g)
        #     print("pred", p)
        #     input()
        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        rouge.update({"gen_len": summ_len})
        return rouge

    def translation_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu

    compute_metrics_fn = summarization_metrics if "summarization" in task_name else translation_metrics
    return compute_metrics_fn


logger = getLogger(__name__)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"setting model.config to task specific params for {task}:\n {pars}")
        logger.info("note: command line args may override some of these")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    try:
        repo = git.Repo(search_parent_directories=True)
        repo_infos = {
            "repo_id": str(repo),
            "repo_sha": str(repo.head.object.hexsha),
            "repo_branch": str(repo.active_branch),
            "hostname": str(socket.gethostname()),
        }
        return repo_infos
    except TypeError:
        return {
            "repo_id": None,
            "repo_sha": None,
            "repo_branch": None,
            "hostname": None,
        }


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict


def calculate_rouge(
    pred_lns: List[str],
    tgt_lns: List[str],
    use_stemmer=True,
    rouge_keys=ROUGE_KEYS,
    return_precision_and_recall=False,
    bootstrap_aggregation=True,
    newline_sep=True,
) -> Dict:
    """Calculate rouge using rouge_scorer package.

    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).

    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys

    """
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lns, pred_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)


# Utilities for freezing parameters and checking whether they are frozen


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def parse_numeric_n_bool_cl_kwargs(unparsed_args: List[str]) -> Dict[str, Union[int, float, bool]]:
    """
    Parse an argv list of unspecified command line args to a dict.
    Assumes all values are either numeric or boolean in the form of true/false.
    """
    result = {}
    assert len(unparsed_args) % 2 == 0, f"got odd number of unparsed args: {unparsed_args}"
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith("--")
        if unparsed_args[i + 1].lower() == "true":
            value = True
        elif unparsed_args[i + 1].lower() == "false":
            value = False
        else:
            try:
                value = int(unparsed_args[i + 1])
            except ValueError:
                value = float(unparsed_args[i + 1])  # this can raise another informative ValueError

        result[unparsed_args[i][2:]] = value
    return result


def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def check_output_dir(args, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it

    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir

    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
        os.path.exists(args.output_dir)
        and len(os.listdir(args.output_dir)) > expected_items
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and "
            f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )
