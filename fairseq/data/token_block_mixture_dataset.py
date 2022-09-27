# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from fairseq.data import FairseqDataset, plasma_utils, data_utils


class TokenBlockMixtureDataset(FairseqDataset):
    """Break a Dataset of tokens into blocks.

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths
        block_sizes (List[int]): maximum block sizes
        include_targets (bool, optional): return next tokens as targets
            (default: False).
        document_sep_len (int, optional): document separator size (required for
            'complete_doc' break mode). Typically 1 if the sentences have eos
            and 0 otherwise.
    """
    def __init__(
        self,
        dataset,
        sizes,
        block_sizes,
        pad,
        eos,
        document_sep_len=1,
    ):
        try:
            from fairseq.data.token_block_utils_fast import (
                _get_slice_indices_fast,
                _get_block_to_dataset_index_fast,
            )
        except ImportError:
            raise ImportError(
                'Please build Cython components with: `pip install --editable .` '
                'or `python setup.py build_ext --inplace`'
            )

        super().__init__()
        self.dataset = dataset
        self.pad = pad
        self.eos = eos

        assert len(dataset) == len(sizes)
        assert len(dataset) > 0

        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            if torch.is_tensor(sizes):
                sizes = sizes.numpy()
            sizes = sizes.astype(np.int64)

        assert min(block_sizes) > 0
        block_sizes = [0] + block_sizes
        slice_indices_list = []
        sizes_list = []
        block_to_dataset_index_list = []
        number_of_inst_in_block = []
        for block_size in block_sizes:
            break_mode = "eos" if block_size == 0 else "complete"
            slice_indices = _get_slice_indices_fast(sizes, break_mode, block_size, document_sep_len)
            slice_indices_list.append(slice_indices)
            sizes_list.append(slice_indices[:, 1] - slice_indices[:, 0])
            number_of_inst_in_block.append(len(slice_indices))

            # build index mapping block indices to the underlying dataset indices
            if break_mode == "eos":
                # much faster version for eos break mode
                block_to_dataset_index = np.stack(
                    [
                        np.arange(len(sizes)),  # starting index in dataset
                        np.zeros(len(sizes), dtype=np.long),  # starting offset within starting index
                        np.arange(len(sizes)),  # ending index in dataset
                    ],
                    1,
                )
            else:
                block_to_dataset_index = _get_block_to_dataset_index_fast(sizes, slice_indices)
            block_to_dataset_index_list.append(block_to_dataset_index)

        self._sizes = np.concatenate(sizes_list)
        self._slice_indices = np.concatenate(slice_indices_list, axis=0)
        self._block_to_dataset_index = np.concatenate(block_to_dataset_index_list, axis=0)
        self._number_of_inst_in_block = np.array(number_of_inst_in_block, dtype=np.int64)

        self._slice_indices = plasma_utils.PlasmaArray(self._slice_indices)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(self._block_to_dataset_index)
        self._number_of_inst_in_block = plasma_utils.PlasmaArray(self._number_of_inst_in_block)

    @property
    def slice_indices(self):
        return self._slice_indices.array

    @property
    def sizes(self):
        return self._sizes.array

    @property
    def block_to_dataset_index(self):
        return self._block_to_dataset_index.array

    @property
    def number_of_inst_in_block(self):
        return self._number_of_inst_in_block.array

    def attr(self, attr: str, index: int):
        start_ds_idx, _, _ = self.block_to_dataset_index[index]
        return self.dataset.attr(attr, start_ds_idx)

    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]

        buffer = torch.cat(
            [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
        )

        slice_s, slice_e = self.slice_indices[index]
        length = slice_e - slice_s
        s, e = start_offset, start_offset + length
        item = buffer[s:e]

        return item

    def num_tokens(self, index):
        return self.sizes[index]

    def __len__(self):
        return len(self.slice_indices)

    def shuffle(self, seed: int):
        with data_utils.numpy_seed(seed):
            bucket_offsets = np.cumsum(self.number_of_inst_in_block) - self.number_of_inst_in_block
            num_buckets = len(self.number_of_inst_in_block)
            shuffled_bucket_indices = np.random.permutation(num_buckets)
            shuffles = []
            for bid in shuffled_bucket_indices:
                shuffle = np.random.permutation(self.number_of_inst_in_block[bid]) + bucket_offsets[bid]
                shuffles.append(shuffle)
            return np.concatenate(shuffles)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(
            {
                ds_idx
                for index in indices
                for start_ds_idx, _, end_ds_idx in [self.block_to_dataset_index[index]]
                for ds_idx in range(start_ds_idx, end_ds_idx + 1)
            }
        )
