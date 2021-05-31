# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import warnings
import json
from tqdm import tqdm

from datasets import load_dataset, load_metric

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
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
)
from transformers import (
    DebertaConfig,
    DebertaTokenizer,
    DebertaForQuestionAnswering,
)
from transformers.trainer_utils import is_main_process
from utils_qa import postprocess_qa_predictions, postprocess_qa_predictions_with_beam_search
from data_utils import (
    proc_dataset,
    load_datasets
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    freeze_emb: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to freeze encoder (if so, we can use large batch size for training)."}
    )
    early_stop: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to do early stopping in the traning process."}
    )
    early_stopping_patience: Optional[int] = field(
        default=1,
        metadata={"help": "`metric_for_best_model` to stop training when the specified metric worsens for `early_stopping_patience` evaluation calls."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    extra_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the extra datasets to use (via the datasets library). split with '|' "}
    )
    extra_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the extra datasets to use (via the datasets library). split with '|' "}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

@dataclass
class ExtraArguments:
    do_ensemble: Optional[bool] = field(
        default=False, metadata={"help": "Whether to do ensemble over multiple predictions"}
    )
    ensemble_file_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the ensemble file."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, ExtraArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, extra_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.freeze_emb:
        for p in [
            *model.roberta.embeddings.parameters()
        ]:
            p.requires_grad = False

    training_args.use_fast = True
    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=False,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            )
        training_args.use_fast = False
        warnings.warn(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models \
            at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this requirement",
            UserWarning
        )

    datasets, extra_datasets = load_datasets(data_args, model_args)

    train_dataset, validation_dataset, question_column_name, context_column_name, answer_column_name = proc_dataset(training_args, data_args, datasets, tokenizer, extra_datasets=extra_datasets)

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = default_data_collator if data_args.pad_to_max_length else DataCollatorWithPadding(tokenizer)

    # Post-processing:
    def post_processing_function(examples, features, predictions):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            is_world_process_zero=trainer.is_world_process_zero(),
        )
        # Format the result to the format the metric expects.
        # if data_args.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
        # else:
        #     formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in datasets["validation"]]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # Initialize early stop callbacks
    callbacks = []
    if model_args.early_stop:
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience)
        callbacks.append(early_stopping_callback)

    # Cache huggingface datasets data and metric files in model_args.cache_dir
    metric = load_metric("squad_v2", cache_dir=model_args.cache_dir)

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)


    if extra_args.do_ensemble:
        assert extra_args.ensemble_file_path is not None
        with open(extra_args.ensemble_file_path, "r") as f:
            spans = json.load(f)
        
        preds = []
        answers = []
        examples = datasets["validation"]
        for example_index, example in enumerate(tqdm(examples)):
            example_id = example["id"]
            if example_id not in spans:
                continue

            context = example["context"]

            span = spans[example_id]
            assert span[0] < span[1] or span[0] == span[1] == 0
            output_text = context[span[0]:span[1]]
            item = {"id": example_id, "prediction_text": output_text, "no_answer_probability":0.0}
            preds.append(item)

            if "test" not in data_args.dataset_config_name:
                answer = {}
                answer["answers"] = example["answers"]
                answer["id"] = example_id
                answers.append(answer)
        
        assert len(preds) == len(spans)
        with open(extra_args.ensemble_file_path.replace(".json", "_preds.json"), "w") as f:
            json.dump(preds, f)

        if "test" not in data_args.dataset_config_name:
            output = metric.compute(predictions=preds, references=answers)
            print(output)
        return None

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
                    
    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
