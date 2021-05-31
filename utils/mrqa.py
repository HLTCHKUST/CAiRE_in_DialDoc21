from __future__ import absolute_import, division, print_function

import json
import logging
import os
import random
from tqdm import tqdm

import datasets

YOUR_LOCAL_DOWNLOAD = "../data"  # For subtask1, Doc2Dial v1.0.1 is already included in the folder "data".
FULL = ["HotpotQA", "NaturalQuestions", "NewsQA", "SQuAD", "SearchQA", "TriviaQA"]
NAMES = ["HotpotQA", "NaturalQuestions", "NewsQA", "SQuAD"] # 
OOD_NAMES = ["DROP", "DuoRC.ParaphraseRC", "RACE", "RelationExtraction", "TextbookQA", "BioASQ"]

_CITATION = """\
    @inproceedings{fisch2019mrqa,
        title={{MRQA} 2019 Shared Task: Evaluating Generalization in Reading Comprehension},
        author={Adam Fisch and Alon Talmor and Robin Jia and Minjoon Seo and Eunsol Choi and Danqi Chen},
        booktitle={Proceedings of 2nd Machine Reading for Reading Comprehension (MRQA) Workshop at EMNLP},
        year={2019},
    }
"""

_DESCRIPTION = """\
    MRQA dataset is a set of multiple dataset, including SQuAD, NewsQA, TriviaQA, SearchQA, HotpotQA, NaturalQuestions.
"""

_HOMEPAGE = "https://github.com/mrqa/MRQA-Shared-Task-2019"


_URLs = "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz" # NewsQA.jsonl.gz, TriviaQA-web.jsonl.gz, SearchQA.jsonl.gz, HotpotQA.jsonl.gz, NaturalQuestionsShort.jsonl.gz


class MRQA(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.2")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="mrqa_rc",
            version=VERSION,
            description="Load MRQA dataset for machine reading comprehension tasks",
        ),
        datasets.BuilderConfig(
            name="mrqa_rc_small",
            version=VERSION,
            description="Load MRQA dataset for machine reading comprehension tasks",
        )
    ]

    DEFAULT_CONFIG_NAME = "mrqa_rc"

    def _info(self):
        if self.config.name == "mrqa_rc" or self.config.name == "mrqa_rc_small":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                            # "spans": datasets.features.Sequence(datasets.Value("string"))
                        }
                    ),
                    "domain": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):

        my_urls = _URLs

        # data_dir = dl_manager.download_and_extract(my_urls) 
        data_dir = YOUR_LOCAL_DOWNLOAD # point to local dir to avoid downloading the dataset again
        if self.config.name == "mrqa_rc" or self.config.name == "mrqa_rc_small":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "mrqa/dev/dataset.jsonl"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "mrqa/train/dataset.jsonl"
                        ),
                    },
                ),
            ]
    
    def _get_answers(self, qa):
        selected_answers = []
        for answer in qa["detected_answers"]:
            selected_answer = {"text": answer["text"], "answer_start": answer["char_spans"][0][0]}
            selected_answers.append(selected_answer)

        if len(selected_answers) > 0:
            return selected_answers 
        else:
            return None

    def _generate_examples(self, filepath):
        """Load dialog data in the reading comprehension task setup, where context is the grounding document,
        input query is dialog history in reversed order, and output to predict is the next agent turn."""
        
        if self.config.name == "mrqa_rc":
            dataset_names = FULL
        elif self.config.name == "mrqa_rc_small":
            dataset_names = NAMES
            

        logging.info("generating examples from = %s", filepath)

        for name in tqdm(dataset_names, total=len(dataset_names), desc='Loading dataset', ncols=100):
            with open(filepath.replace("dataset", name), encoding="utf-8") as f:
                header = next(f)
                domain = json.loads(header)["header"]["dataset"]

                for row in f:
                    line = json.loads(row)

                    if "<Table>" in line["context"] or "<Tr>" in line["context"] or "<Td>" in line["context"] or "<Th>" in line["context"]:  # filter out table data in NaturalQuestions
                        continue
                    context = line["context"].strip().replace("\n\n\n\n", "\t").replace("\n", "\t") # for NewsQA
                    context = context.replace("[PAR]", "").replace("[TLE]", "").replace("[SEP]", "") # for HotpotQA
                    context = context.replace("<P>", "").replace("</P>", "") # for NaturalQuestions

                    for qa in line["qas"]:
                        qid = qa["qid"]
                        question = qa["question"].strip()
                        detected_answers = []
                        for detect_ans in qa["detected_answers"]:
                            detected_answers.append(
                                {
                                    "text": detect_ans["text"].strip(),
                                    "answer_start": detect_ans["char_spans"][0][0]
                                }
                            )
                        yield qid, {
                            "domain": domain,
                            "context": context,
                            "id": qid,
                            "title": qid,
                            "question": question,
                            "answers": detected_answers,
                        }


                            
