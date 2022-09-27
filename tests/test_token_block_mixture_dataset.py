# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from fairseq.data import TokenBlockMixtureDataset, SortDataset, data_utils

import tests.utils as test_utils


class TestTokenBlockDataset(unittest.TestCase):

    def _build_dataset(self, data, **kwargs):
        sizes = [len(x) for x in data]
        underlying_ds = test_utils.TestDataset(data)
        return TokenBlockMixtureDataset(underlying_ds, sizes, **kwargs)

    def test_complete_break_mode(self):
        data = [
            torch.tensor([3, 4, 1], dtype=torch.long),
            torch.tensor([5, 4, 3, 2, 1], dtype=torch.long),
            torch.tensor([8, 7, 6, 1], dtype=torch.long),
            torch.tensor([9, 1], dtype=torch.long),
            torch.tensor([4, 1], dtype=torch.long),
            torch.tensor([3, 4, 6, 7, 8, 1], dtype=torch.long),
        ]
        ds = self._build_dataset(data, block_sizes=[4, 8, 16], pad=0, eos=1)
        print(len(ds))
        for i in range(len(ds)):
            print(i, ds[i])
        self.assertEqual(ds[0].tolist(), [3, 4, 1])
        self.assertEqual(ds[1].tolist(), [5, 4, 3, 2, 1])

        print(ds.number_of_inst_in_block)
        with data_utils.numpy_seed(1):
            shuffle = np.random.permutation(len(ds))
        print(ds.sizes)
        print(shuffle)
        sds = SortDataset(ds, sort_order=[shuffle, ds.sizes])
        print(sds.ordered_indices())

if __name__ == "__main__":
    unittest.main()
