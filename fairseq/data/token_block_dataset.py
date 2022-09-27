# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from fairseq.data import FairseqDataset, plasma_utils

from fairseq.data.token_block_utils_fast import (
    _get_slice_indices_fast,
    _get_block_to_dataset_index_fast,
)


class TokenBlockDataset(FairseqDataset):
    """Break a Dataset of tokens into blocks.

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
        block_size (int): maximum block size (ignored in 'eos' break mode)
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
            - 'complete': break tokens into blocks (up to block_size) such that
                blocks contains complete sentences, although block_size may be
                exceeded if some sentences exceed block_size
            - 'complete_doc': similar to 'complete' mode, but do not
                cross document boundaries
            - 'eos': each block contains one sentence (block_size is ignored)
        include_targets (bool, optional): return next tokens as targets
            (default: False).
        document_sep_len (int, optional): document separator size (required for
            'complete_doc' break mode). Typically 1 if the sentences have eos
            and 0 otherwise.
        variant_block_size_multiples (tuple, optional): each chunk is e.g. [3, 6] times of a block size
    """
    def __init__(
        self,
        dataset,
        sizes,
        block_size,
        pad,
        eos,
        break_mode=None,
        include_targets=False,
        document_sep_len=1,
        variant_block_size_multiples=None,
        seed=0,
    ):
        # try:
        #     from fairseq.data.token_block_utils_fast import (
        #         _get_slice_indices_fast,
        #         _get_block_to_dataset_index_fast,
        #     )
        # except ImportError:
        #     raise ImportError(
        #         'Please build Cython components with: `pip install --editable .` '
        #         'or `python setup.py build_ext --inplace`'
        #     )

        super().__init__()
        self.dataset = dataset
        self.pad = pad
        self.eos = eos
        self.include_targets = include_targets

        assert len(dataset) == len(sizes)
        assert len(dataset) > 0

        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            if torch.is_tensor(sizes):
                sizes = sizes.numpy()
            sizes = sizes.astype(np.int64)

        if variant_block_size_multiples is not None:
            self.variant_block_multiple_min, self.variant_block_multiple_max = variant_block_size_multiples[0], \
                                                                    variant_block_size_multiples[1]
        else:
            self.variant_block_multiple_min, self.variant_block_multiple_max = 1, 1

        break_mode = break_mode if break_mode is not None else 'none'

        # For "eos" break-mode, block_size is not required parameters.
        if break_mode == "eos" and block_size is None:
            block_size = 0

        self.input_data_sizes = sizes
        self.break_mode = break_mode
        self.block_size = block_size
        self.document_sep_len = document_sep_len
        self._cur_epoch = 1
        self.seed = seed

        if self.variant_block_multiple_max > 1:
            block_sizes = [i * block_size for i in range(variant_block_size_multiples[0], variant_block_size_multiples[1] + 1)]
            k = len(sizes)
            blocks = list(np.random.choice(block_sizes, k))
        else:
            blocks = list(np.ones(len(sizes)) * block_size)
        slice_indices = _get_slice_indices_fast(sizes, break_mode, block_size, document_sep_len,
                                                self.variant_block_multiple_min, self.variant_block_multiple_max, blocks)
        self._sizes = slice_indices[:, 1] - slice_indices[:, 0]

        # build index mapping block indices to the underlying dataset indices
        if break_mode == "eos":
            # much faster version for eos break mode
            block_to_dataset_index = np.stack(
                [
                    np.arange(len(sizes)),  # starting index in dataset
                    np.zeros(
                        len(sizes), dtype=np.long
                    ),  # starting offset within starting index
                    np.arange(len(sizes)),  # ending index in dataset
                ],
                1,
            )
        else:
            block_to_dataset_index = _get_block_to_dataset_index_fast(
                sizes,
                slice_indices,
            )
        # max document size for padding, only used for mega LM at inference time
        self.max_example_size = max(self._sizes)

        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(block_to_dataset_index)

    def reindex(self, epoch):
        if epoch == self._cur_epoch:
            return

        self._cur_epoch = epoch
        rng = np.random.RandomState(
            [
                42,  # magic number
                self.seed % (2 ** 32),  # global seed
                self._cur_epoch,  # epoch index
            ]
        )
        block_sizes = [i * self.block_size for i in
                       range(self.variant_block_multiple_min, self.variant_block_multiple_max + 1)]
        k = len(self.dataset) // 2
        blocks = list(rng.choice(block_sizes, k))
        slice_indices = _get_slice_indices_fast(self.input_data_sizes, self.break_mode, self.block_size, self.document_sep_len,
                                                self.variant_block_multiple_min, self.variant_block_multiple_max, blocks)
        self._sizes = slice_indices[:, 1] - slice_indices[:, 0]

        # build index mapping block indices to the underlying dataset indices
        block_to_dataset_index = _get_block_to_dataset_index_fast(
            self.input_data_sizes,
            slice_indices,
        )
        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(block_to_dataset_index)

    @property
    def slice_indices(self):
        return self._slice_indices.array

    @property
    def sizes(self):
        return self._sizes.array

    @property
    def block_to_dataset_index(self):
        return self._block_to_dataset_index.array

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

        if self.include_targets:
            # *target* is the original sentence (=item)
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            if s == 0:
                source = torch.cat([item.new([self.eos]), buffer[0 : e - 1]])
                past_target = torch.cat(
                    [item.new([self.pad, self.eos]), buffer[0 : e - 2]]
                )
            else:
                source = buffer[s - 1 : e - 1]
                if s == 1:
                    past_target = torch.cat([item.new([self.eos]), buffer[0 : e - 2]])
                else:
                    past_target = buffer[s - 2 : e - 2]

            return source, item, past_target

        return item

    def __len__(self):
        return len(self.slice_indices)

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
