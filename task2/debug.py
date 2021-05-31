import json
import linecache
import math
import os
import pickle
import socket
import numpy as np
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt 

import transformers
from transformers import AutoTokenizer

src_file = "data/grounding/val_grounding.source"
tgt_file = "data/grounding/val_grounding.target"


def get_char_lens(data_file):
    return [len(x) for x in Path(data_file).open().readlines()]

def get_tokenized_lens(tokenizer, lines):
    return [len(tokenizer.encode(x)) for x in lines]

def get_line(data_file, index):
    return linecache.getline(str(data_file), index+1).rstrip("\n")

src_len = get_char_lens(src_file)
N = len(src_len)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

source_lines = []
target_lines = []
for idx in range(N):
    source_line = get_line(src_file, idx)
    target_line = get_line(tgt_file, idx)
    source_lines.append(source_line)
    target_lines.append(target_line)

slens = get_tokenized_lens(tokenizer, source_lines)
print(max(slens), min(slens), np.mean(slens))
tlens = get_tokenized_lens(tokenizer, target_lines)
print(max(tlens), min(tlens), np.mean(tlens))
plt.hist(slens)
plt.savefig('slen.png')

plt.hist(tlens)
plt.savefig('tlen.png')