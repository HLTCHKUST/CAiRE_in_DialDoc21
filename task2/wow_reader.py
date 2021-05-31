"""
Compare with the previous dialog_reader, this dialog_reader removes the token type id 
but add padding token and other corresponding special tokens instead.
"""
import os
import json
import pickle
import random
import pprint
import logging
import numpy as np
from tqdm import tqdm
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

from parlai.core.dict import DictionaryAgent
from parlai.core.worlds import create_task

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer

from src.data_utils.utils import pad_sents, get_mask, pad_list_of_sents, get_list_of_mask
from src.data_utils.data_reader import getDataLoader

class DialogReader(Dataset):
    def __init__(self,
                 tokenizer,
                 mode: str = "train",
                 max_length: int = 128,
                 max_context_length: int = 128,
                 max_kn_length: int = 128,
                 max_episode_length: int = 1,
                 data_dir: str = "./data",
                 history_in_context: bool = False,
                 kn_in_context: bool = False,
                 model_type: str = "decoder_only",
                 debug: bool = False,
                 inference: bool = False, 
                 kn_mode: str = "oracle",
                 ):
        self._max_length = max_length
        self._max_context_length = max_context_length
        self._max_kn_length = max_kn_length
        self._max_episode_length = max_episode_length
        self._data_dir = data_dir
        self._model_type = model_type
        self._kn_mode = kn_mode

        self._tokenizer = tokenizer
        self._debug = debug
        self._inference = inference

        self.data = self.read(mode, history_in_context, kn_in_context)

    def __getitem__(self, idx):
        """Returns one data pair (source and target)."""
        item = {}
        for key in ["context", "response", "chosen_sentence"]:
            item[key] = torch.LongTensor(self.data[key][idx])
            item[f"{key}_mask"] = torch.LongTensor(self.data[f"{key}_mask"][idx])
        
        return item

    def __len__(self):
        return len(self.data["episode_id"])

    def _load_and_preprocess_all(self, mode: str):
        raise NotImplementedError

    def read(self, mode: str, history_in_context: bool, kn_in_context: bool):
        def _gen(episodes, sos_token, eos_token):
            """
            Convert example into samples for training and testing
            1. truncate the knowledge
            2. split the episode into training samples
            """
            samples = {
                "context": [],             # list
                "response": [],            # list
                "chosen_sentence": [],     # list
            }

            if self._model_type == "decoder_only":
                samples.update({"token_type":[]})

            for _id, episode in enumerate(tqdm(episodes, desc="Generate samples", ncols=100)):
                contexts = episode["context"]                          # list
                response = episode["response"]                         # list
                checked_sentence = episode["checked_sentence"]
                episode_num = episode["episode_num"]                   # int
                episode_length = len(episode["context"])

                token_type_ids = []

                # get knowledge
                kn = self._tokenizer.encode(checked_sentence.strip(), add_special_tokens=False) + self._tokenizer.encode("<eos_k>", add_special_tokens=False)
                if kn_in_context:
                    token_type_ids += [0] * len(kn)
                
                # get history
                # USR_{t-1} <eos_u> SYS_{t-1} <eos_r> USR_{t} <eos_u>
                history = []
                if history_in_context:
                    for num in range(len(contexts)):
                        temp = self._tokenizer.encode(contexts[num].strip(), add_special_tokens=False)
                        history += temp
                        if num % 2 == 0:
                            history += self._tokenizer.encode("<eos_u>", add_special_tokens=False)
                            token_type_ids += [1] * len(temp)
                        else:
                            history += self._tokenizer.encode("<eos_r>", add_special_tokens=False)
                            token_type_ids += [0] * len(temp)
                    if kn_in_context:
                        context = kn + history
                    else:
                        context = history
                else:
                    context =  self._tokenizer.encode(contexts[-1].strip()+"<eos_u>",  add_special_tokens=False) 

                if self._model_type == "seq2seq": # [bos] seq1 [sep] seq2 [eos]
                    samples["context"].append(self._tokenizer.build_inputs_with_special_tokens(context))
                    samples["response"].append(self._tokenizer.encode(response.strip()))
                elif self._model_type == "decoder_only":  # GPT2 model without pre-defined special tokens
                    context = self._tokenizer.encode(context)  
                    response = self._tokenizer.encode(sos_token+response.strip()+eos_token) 
                    response_mask = [1] * (len(context)+1) + [0] * (len(response)-1)
                    token_type_ids = [1] * len(response)
                    context += response
                    samples["context"].append(context)
                    samples["response"].append(response_mask)
                else:
                    raise ValueError(f"Model type '{self._model_type}' is invalid.")

                samples["chosen_sentence"].append(self._tokenizer.encode(checked_sentence.strip()))

                if self._model_type == "decoder_only":
                    samples["token_type"].append(token_type_ids)

                if self._debug:
                    if len(samples["context"]) >= 10:
                        break
            return samples

        def _uniform(samples, history_in_context):
            """
            1. pad the sents in the same sample to the maximum length
            2. get the length of the inputs
            3. get the mask of the inputs
            """
            uniformed_samples = {}
            for key in samples:
                # Get sample data, check the type
                sample = samples[key]
                pad_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer._pad_token)
                if key == "checked_sentence":
                    mask = get_mask(sample, max_len=self._max_kn_length)
                    padded_sample, _ = pad_sents(sample, pad_token=pad_id, max_len=self._max_kn_length)
                elif key == "context" or key == "token_type":
                    mask = get_mask(sample, max_len=self._max_context_length)
                    padded_sample, _ = pad_sents(sample, pad_token=pad_id, max_len=self._max_context_length)
                elif "kn" in key:
                    mask = get_mask(sample, max_len=1024)
                    padded_sample, _ = pad_sents(sample, pad_token=pad_id, max_len=1024)
                else:
                    mask = get_mask(sample, max_len=self._max_length)
                    padded_sample, _ = pad_sents(sample, pad_token=pad_id, max_len=self._max_length)
                uniformed_samples[key] = padded_sample
                uniformed_samples[f"{key}_mask"] = mask
            return uniformed_samples

        # read the datasets and do preprocessing (tokenize, set up KN)
        episodes = self._load_and_preprocess_all(mode, history_in_context, self._max_episode_length)
        # formulate the samples
        sos_token = self._tokenizer.bos_token
        eos_token = self._tokenizer.eos_token

        samples = _gen(episodes, sos_token, eos_token)
        uniformed_samples = _uniform(samples, history_in_context)
        return uniformed_samples

    @property
    def vocab(self):
        return self._tokenizer.vocab

    def _preprocess_episodes(self, episodes, mode, history_in_context, max_episode_length=1):
        """
        Tokenize all the fields in Wizard-of-Wikipedia.
        Return List[Dict[samples](episodes)]

        Output Example:
        [
            { # one episode
                'context': [], # in episode length
                'response': [],
                'title': [],
                'sample_id': int,
                'episode_num': int,
            }
            ...
            {
                # another episode
            }
        ]
        """
        new_episodes = []
        for episode_num, episode in enumerate(tqdm(episodes, desc="Preprocess episodes", ncols=100)):
            for example_num, example in enumerate(episode):        
                new_examples = {'context': [],
                                'response': '',
                                'checked_sentence': ''}

                history = []
                if example_num != 0 and history_in_context:
                    start_idx = max(0, example_num-max_episode_length)
                    for num in range(start_idx, example_num):
                        history.append(episode[num]['text'].lower().strip())
                        history.append(episode[num]['labels'][0].lower().strip() if mode == "train" else episode[num]['eval_labels'][0].lower().strip())
                context = history + [example['text'].lower().strip()]
                
                if mode == "train":
                    response = example['labels'][0]
                else:
                    response = example['eval_labels'][0]
                checked_sentence = example['checked_sentence']

                new_examples['context'] = context
                new_examples['response'] = response.lower() 
                new_examples['checked_sentence'] = checked_sentence.lower()

                new_episodes.append(new_examples)

        return new_episodes


class WowDialogReader(DialogReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._task = "wizard_of_wikipedia"

    def _load_and_preprocess_all(self, mode: str, history_in_context: bool, max_episode_length: int):
        """
        As default, it returns the following action dict:
        {
            'id': 'wizard_of_wikipedia'
            'text': chosen_topic\n # if first example in episode
                    last_apprentice_message\n # if possible
                    wizard_message # if --label-type is 'chosen_sent'
            'knowledge': title_1 sentence_1\n
                                .
                                .
                                .
                         title_m sentence_n # all knowledge available to wizard
            'labels': [title_checked sentence_checked] # default
                                    OR
                      [wizard_response] # if --label-type set to 'response'
            'label_candidates': knowledge + [no_passages_used no_passages_used]
                                           OR
                                100 response candidates  # if 'validation' or 'test'
            'chosen_topic': chosen_topic as untokenized string
            'checked_sentence': checked sentence if wizard, else None # if --include_checked_sentence
            'title': title of checked sentence # if --include_checked_sentence
            --> if not exists, then checked_sentence = title = 'no_passages_used'
            'episode_done': (Boolean) whether episode is done or not
        }
        """
        parlai_opt = self._get_parlai_opt([
            '--task', 'wizard_of_wikipedia:generator:topic_split' if 'unseen' in mode else 'wizard_of_wikipedia:generator:random_split',
            '--datatype', '{}:stream'.format(mode.split('_')[0]) if 'unseen' in mode else f'{mode}:stream',  # 'train' for shuffled data and 'train:stream' for unshuffled data
            '--datapath', self._data_dir,
            '--include_knowledge_separator', 'True',  # include speical __knowledge__ token between title and passage
            '--include_checked_sentence', 'True',
            '--label_type', 'response', # choices = ['response', 'chosen_sent']
        ])
        # As a default, world use "WizardDialogKnowledgeTeacher"
        agent = DictionaryAgent(parlai_opt)
        world = create_task(parlai_opt, agent)
        num_examples = world.num_examples()
        num_episodes = world.num_episodes()

        episodes = []
        for _ in range(num_episodes):
            examples = []
            while True:
                world.parley()
                example = world.acts[0]
                examples.append(example)
                if world.episode_done():
                    episodes.append(examples)
                    break

        return self._preprocess_episodes(episodes, mode, history_in_context, max_episode_length=max_episode_length)

    def _get_parlai_opt(self, options: List[str] = []):
        from parlai.scripts.build_dict import setup_args
        parser = setup_args()
        opt = parser.parse_args(options)
        return opt

def get_wow_dataloader(args, tokenizer, train=True, shuffle_train=True):
    if train:
        train_reader = WowDialogReader(
            tokenizer,
            mode="train",
            max_length = args.max_length,
            max_context_length = args.max_context_length,
            max_kn_length = args.max_kn_length,
            max_episode_length = args.max_episode_length,
            data_dir = args.data_dir,
            history_in_context = args.history_in_context,
            kn_in_context = args.kn_in_context,
            model_type = args.model_type,
            debug = args.debug if hasattr(args, 'debug') else False,
            inference = args.inference if hasattr(args, 'inference') else False,
        )
        train_loader = getDataLoader(train_reader, args.bsz, test=False if shuffle_train else True)
    else:
        train_loader = None

    valid_reader = WowDialogReader(
        tokenizer,
        mode="valid",
        max_length = args.max_length,
        max_context_length = args.max_context_length,
        max_kn_length = args.max_kn_length,
        max_episode_length = args.max_episode_length,
        data_dir = args.data_dir,
        history_in_context = args.history_in_context,
        kn_in_context = args.kn_in_context,
        model_type = args.model_type,
        debug = args.debug if hasattr(args, 'debug') else False,
        inference = args.inference if hasattr(args, 'inference') else False,\
    )
    valid_loader = getDataLoader(valid_reader, args.eval_bsz, test=True)

    valid_unseen_reader = WowDialogReader(
        tokenizer,
        mode="valid_unseen",
        max_length = args.max_length,
        max_context_length = args.max_context_length,
        max_kn_length = args.max_kn_length,
        max_episode_length = args.max_episode_length,
        data_dir = args.data_dir,
        history_in_context = args.history_in_context,
        kn_in_context = args.kn_in_context,
        model_type = args.model_type,
        debug = args.debug if hasattr(args, 'debug') else False,
        inference = args.inference if hasattr(args, 'inference') else False,
    )
    valid_unseen_loader = getDataLoader(valid_unseen_reader, args.eval_bsz, test=True)

    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
        "valid_unseen": valid_unseen_loader,
    }
    return dataloaders

def get_data_from_batch(batch, model_type="decoder_only"):
    kn_sent = batch["chosen_sentence"]
    kn_mask = batch["chosen_sentence_mask"]

    if model_type == "seq2seq":
        inputs = batch["context"]
        masks = batch["context_mask"]
        labels = batch["response"]
        label_masks = batch["response_mask"]
        response_masks = None

        label_starts = torch.Tensor([0]*inputs.size(0))
        label_idxs = torch.sum(label_masks, 1)

    else:
        seqlen = batch["context"].size(1)
        inputs = batch["context"].narrow(1, 0, seqlen - 1).clone() # 
        masks = batch["context_mask"].narrow(1, 0, seqlen - 1).clone() # 
        labels = batch["context"].narrow(1, 1, seqlen-1).clone() # 
        label_masks = batch["context_mask"].narrow(1, 1, seqlen - 1).clone() # 
        response_masks = batch["response"].narrow(1, 1, seqlen - 1).clone() # 

        label_starts = torch.sum(response_masks, 1)
        label_idxs = torch.sum(label_masks, 1)

    return inputs, masks, kn_sent, kn_mask, topic, topic_masks, \
        labels, label_masks, response_masks, label_starts, label_idxs, None

if __name__ == "__main__":
    args = {
        "max_length" : 128,  
        "max_context_length" : 128,   # 256
        "max_kn_length" : 128,
        "max_episode_length" : 1,
        "data_dir" : "./data",
        "model": "facebook/bart-base",
        "model_type" : "seq2seq",
        "bsz" : 3,
        "history_in_context" : True,
        "kn_in_context" : True, 

    }
    tokenizer = AutoTokenizer.from_pretrained(args["model"])
    reader = WowDialogReader(
        tokenizer,
        mode="valid",
        max_length = args["max_length"],
        max_kn_length = args["max_kn_length"],
        max_episode_length = args["max_episode_length"],
        data_dir = args["data_dir"],
        model_tyep = args["model_type"],
        history_in_context = args["history_in_context"],
        kn_in_context = args["kn_in_context"],
    )
