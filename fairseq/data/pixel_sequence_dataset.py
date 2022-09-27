# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
import os

import numpy as np
import torch

from . import FairseqDataset
from fairseq.tokenizer import tokenize_line


class PixelSequenceDataset(FairseqDataset):

    def __init__(self, path, normalization, reverse_order=False):
        self.mean = normalization[0]
        self.std = normalization[1]
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.reverse_order = reverse_order
        self.read_data(path)
        self.size = len(self.tokens_list)

    def read_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                tokens = self.encode_line(line, reverse_order=self.reverse_order)
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def encode_line(self, line, line_tokenizer=tokenize_line, reverse_order=False):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        pixels = [int(w) for w in words]

        default_float_dtype = torch.get_default_dtype()
        pixels = torch.tensor(pixels, dtype=default_float_dtype).div(255.)
        return pixels.sub(self.mean).div(self.std)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)
