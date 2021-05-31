import logging
import os
import sys
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
# from multiprocessing import cpu_count

from datasets import load_dataset, load_metric, concatenate_datasets

import transformers
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    squad_convert_examples_to_features,
    SquadExample,
    SquadFeatures,
    DataProcessor,
)
from transformers.trainer_utils import is_main_process
from utils_qa import postprocess_qa_predictions


logger = logging.getLogger(__name__)

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)



def load_datasets(data_args, model_args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, field="data")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    extra_datasets = []
    if data_args.extra_dataset_name is not None:
        dataset_names = data_args.extra_dataset_name.split('|')
        dataset_config_names = data_args.extra_dataset_config_name.split('|')
        for dataset_name, dataset_config_name in zip(dataset_names, dataset_config_names):
            extra_dataset = load_dataset(dataset_name, dataset_config_name, cache_dir=model_args.cache_dir)
            extra_datasets.append(extra_dataset)

    return datasets, extra_datasets



def proc_dataset(training_args, data_args, datasets, tokenizer, extra_datasets=[]):
    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if not training_args.use_fast:
        processor = DataProcessor(datasets)

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        if training_args.use_fast:
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=data_args.max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )
            # Let's label those examples!
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []
            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]
                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    # Start token index of the current span in the text.
                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1

                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index)
            return tokenized_examples
        else:
            # if tokenizer is not TokenizerFast (e.g., for DeBERTa), we have to go back to the old-school preprocessing
            # print('currently not supporting slow tokenizers')
            # raise NotImplementedError
            # it should look like this (similar to the one previously)
            examples = processor.get_samples("train")
            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer, 
                max_seq_length=data_args.max_seq_length,
                doc_stride=data_args.doc_stride,
                max_query_length=data_args.max_seq_length,
                is_training=True,
                padding_strategy="max_length" if data_args.pad_to_max_length else False,
                return_dataset="pt",
            )
            return dataset

    if training_args.do_train:
        if len(extra_datasets) > 0:
            extra_ds = []
            for d in extra_datasets:
                extra_ds.append(d["train"])
            full_dataset = concatenate_datasets([datasets["train"]]+extra_ds)
        else:
            full_dataset = datasets["train"]

        train_dataset = full_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        train_dataset = None

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        if training_args.use_fast:
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=data_args.max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )
            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
            # corresponding example_id and we will store the offset mappings.
            tokenized_examples["example_id"] = []

            for i in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right else 0

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

            return tokenized_examples
        else:
            examples = processor.get_samples("validation")
            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer, 
                max_seq_length=data_args.max_seq_length,
                doc_stride=data_args.doc_stride,
                max_query_length=data_args.max_seq_length,
                is_training=True,
                padding_strategy="max_length" if data_args.pad_to_max_length else False,
                return_dataset="pt",
            )
            return dataset

    if training_args.do_eval:
        validation_dataset = datasets["validation"].map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        validation_dataset = None
    return train_dataset, validation_dataset, question_column_name, context_column_name, answer_column_name


class DialDocProcessor(DataProcessor):
    def __init__(self, datasets=None, dataset_name=None, dataset_config_name=None, cache_dir=None, version_2_with_negative=True):
        if datasets is None:
            assert (dataset_name is not None) and (dataset_config_name is not None)
            datasets = self._load_dataset(dataset_name, dataset_config_name, cache_dir)

        self.datasets = datasets
    
    def _load_dataset(self, dataset_name, dataset_config_name, cache_dir):
        datasets = load_dataset(dataset_name, dataset_config_name, cache_dir=cache_dir)
        return datasets
    
    def get_examples(self, split):
        examples = []
        if split in ['train', 'validation']:
            data = self.datasets[split]
            for _id, title, context, question, answers, domain in tqdm(zip(data["id"], data["title"], data["context"], data["question"], data["answers"], data["domain"]), total=len(data["id"]), ncols=100):
                answer = [{'text':answers["text"][0], 'start_position_character': answers["answer_start"][0]}]
                example = SquadExample(
                    qas_id=_id,
                    question_text=question,
                    context_text=context,
                    answer_text=answers["text"],
                    start_position_character=answers["answer_start"][0],
                    title=title,
                    is_impossible=False,
                    answers=answer,
                )
                examples.append(example)
        else:
            data = self.datasets['validation']
            for _id, title, context, question, domain in tqdm(zip(data["id"], data["title"], data["context"], data["question"], data["domain"]), total=len(data["id"]), ncols=100):
                example = SquadExample(
                    qas_id=_id,
                    question_text=question,
                    context_text=context,
                    title=title,
                    is_impossible=False,
                    answer_text='',
                    start_position_character=None,
                )
                examples.append(example)
        return examples