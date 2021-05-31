# Lint as: python3
"""CoQA: A Conversational Question Answering Challenge"""
# partially taken from https://github.com/NTU-SQUAD/transformers-coqa/blob/2dfd58b70956e935e370989fa421f34bb83bff08/data/processors/coqa.py

from __future__ import absolute_import, division, print_function

import json
import logging
import os
import re
import string
from collections import Counter

import spacy
import datasets

MAX_Q_LEN = 100  # Max length of question
YOUR_LOCAL_DOWNLOAD = "../data"  

_CITATION = """\
    @article{reddy-etal-2019-coqa,
        title = "{C}o{QA}: A Conversational Question Answering Challenge",
        author = "Reddy, Siva  and
        Chen, Danqi  and
        Manning, Christopher D.",
        journal = "Transactions of the Association for Computational Linguistics",
        volume = "7",
        month = mar,
        year = "2019",
        url = "https://www.aclweb.org/anthology/Q19-1016",
        doi = "10.1162/tacl_a_00266",
        pages = "249--266",
        }
"""

_DESCRIPTION = """\
    CoQA is a large-scale dataset for building Conversational Question Answering systems. \
    The goal of the CoQA challenge is to measure the ability of machines to understand a text passage\
    and answer a series of interconnected questions that appear in a conversation. \
    CoQA is pronounced as coca.
"""

_HOMEPAGE = "https://stanfordnlp.github.io/coqa/"


_URLs = "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json, https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json"


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


class Coqa(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="coqa_rc",
            version=VERSION,
            description="Load CoQA dataset for machine reading comprehension tasks",
        )
    ]

    DEFAULT_CONFIG_NAME = "coqa_rc"

    def _info(self):
        if self.config.name == "coqa_rc":
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
                            # "answer_end": datasets.Value("int32"),
                        }
                    ),
                    "domain": datasets.Value("string"),  # is "source" in CoQA (e.g., wikipedia)
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
        if self.config.name == "coqa_rc":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "coqa/coqa-dev-v1.0.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "coqa/coqa-train-v1.0.json"
                        ),
                    },
                ),
            ]

    def _str(self, s):
        """ Convert PTB tokens to normal tokens """
        if (s.lower() == '-lrb-'):
            s = '('
        elif (s.lower() == '-rrb-'):
            s = ')'
        elif (s.lower() == '-lsb-'):
            s = '['
        elif (s.lower() == '-rsb-'):
            s = ']'
        elif (s.lower() == '-lcb-'):
            s = '{'
        elif (s.lower() == '-rcb-'):
            s = '}'
        return s

    def space_extend(self, matchobj):
        return ' ' + matchobj.group(0) + ' '

    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    
    def pre_proc(self, text):
        text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t', self.space_extend, text)
        text = text.strip(' \n')
        text = re.sub('\s+', ' ', text)
        return text

    def process(self, parsed_text):
        output = {'word': [], 'offsets': [], 'sentences': []}

        for token in parsed_text:
            output['word'].append(self._str(token.text))
            output['offsets'].append((token.idx, token.idx + len(token.text)))

        word_idx = 0
        for sent in parsed_text.sents:
            output['sentences'].append((word_idx, word_idx + len(sent)))
            word_idx += len(sent)

        assert word_idx == len(output['word'])
        return output
        
    def normalize_answer(self, s):
        """Lower text and remove punctuation, storys and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_raw_context_offsets(self, words, raw_text):
        raw_context_offsets = []
        p = 0
        for token in words:
            while p < len(raw_text) and re.match('\s', raw_text[p]):
                p += 1
            if raw_text[p:p + len(token)] != token:
                print('something is wrong! token', token, 'raw_text:',
                      raw_text)

            raw_context_offsets.append((p, p + len(token)))
            p += len(token)

        return raw_context_offsets

    def find_span(self, offsets, start, end):
        start_index = -1
        end_index = -1
        for i, offset in enumerate(offsets):
            if (start_index < 0) or (start >= offset[0]):
                start_index = i
            if (end_index < 0) and (end <= offset[1]):
                end_index = i
        return (start_index, end_index)
    
    def find_span_with_gt(self, context, offsets, ground_truth):
        best_f1 = 0.0
        best_span = (len(offsets) - 1, len(offsets) - 1)
        gt = self.normalize_answer(self.pre_proc(ground_truth)).split()

        ls = [
            i for i in range(len(offsets))
            if context[offsets[i][0]:offsets[i][1]].lower() in gt
        ]

        for i in range(len(ls)):
            for j in range(i, len(ls)):
                pred = self.normalize_answer(
                    self.pre_proc(
                        context[offsets[ls[i]][0]:offsets[ls[j]][1]])).split()
                common = Counter(pred) & Counter(gt)
                num_same = sum(common.values())
                if num_same > 0:
                    precision = 1.0 * num_same / len(pred)
                    recall = 1.0 * num_same / len(gt)
                    f1 = (2 * precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_span = (ls[i], ls[j])
        return best_span

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)["data"]
        
        for row in data:
            story = row["story"]
            domain = row["source"]
            title = row["filename"]
            all_prev_utterances = []
            for i, question in enumerate(row['questions']):
                id_ = str(row['id']) + '_' + str(question['turn_id'])
                all_prev_utterances.append(question['input_text'])
                answers =  [{
                    "text": row["answers"][i]["span_text"],
                    "answer_start": row["answers"][i]["span_start"]
                }]

                question_str = " ".join(
                                    list(reversed(all_prev_utterances))
                                ).strip()
                question_str = " ".join(question_str.split()[:MAX_Q_LEN])
                    
                # append the original answer into the utterance list
                orig_answer_text = row["answers"][i]["input_text"]
                all_prev_utterances.append(orig_answer_text)

                qa = {
                        "id": id_,
                        "domain": domain,
                        "title": title,
                        "context": story,
                        "question": question_str,
                        "answers": answers,
                }
                yield id_, qa