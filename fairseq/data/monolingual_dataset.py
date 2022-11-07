# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)
from . import data_utils, FairseqDataset


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key, is_list=False):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(data_utils.collate_tokens(
                    [s[key][i] for s in samples], pad_idx, eos_idx, left_pad=False,
                ))
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples], pad_idx, eos_idx, left_pad=False,
            )

    src_tokens = merge('source')
    if samples[0]['target'] is not None:
        is_target_list = isinstance(samples[0]['target'], list)
        target = merge('target', is_target_list)
    else:
        target = src_tokens

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'nsentences': len(samples),
        'ntokens': sum(len(s['source']) for s in samples),
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': torch.LongTensor([
                s['source'].numel() for s in samples
            ]),
        },
        'target': target,
    }


class MonolingualDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching
            (default: True).
    """

    def __init__(self, dataset, sizes, src_vocab, tgt_vocab, add_eos_for_other_targets, shuffle,
                 targets=None, add_bos_token=False, pad_to_a_length=-1):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        self.add_bos_token = add_bos_token

        assert targets is None or all(t in {'self', 'future', 'past'} for t in targets), \
            "targets must be none or one of 'self', 'future', 'past'"
        if targets is not None and len(targets) == 0:
            targets = None
        self.targets = targets
        self.pad_to_a_length = pad_to_a_length

    def __getitem__(self, index):
        if self.targets is not None:
            # *future_target* is the original sentence
            # *source* is shifted right by 1 (maybe left-padded with eos)
            # *past_target* is shifted right by 2 (left-padded as needed)
            #
            # Left-to-right language models should condition on *source* and
            # predict *future_target*.
            # Right-to-left language models should condition on *source* and
            # predict *past_target*.
            source, future_target, past_target = self.dataset[index]
            source, target = self._make_source_target(source, future_target, past_target)
        else:
            source = self.dataset[index]
            target = None
        source, target = self._maybe_add_bos(source, target)
        source, target = self._maybe_pad(source, target)
        return {'id': index, 'source': source, 'target': target}

    def __len__(self):
        return len(self.dataset)

    def _make_source_target(self, source, future_target, past_target):
        if self.targets is not None:
            target = []

            if self.add_eos_for_other_targets and (('self' in self.targets) or ('past' in self.targets)) \
                    and source[-1] != self.vocab.eos():
                # append eos at the end of source
                source = torch.cat([source, source.new([self.vocab.eos()])])

                if 'future' in self.targets:
                    future_target = torch.cat([future_target, future_target.new([self.vocab.pad()])])
                if 'past' in self.targets:
                    # first token is before the start of sentence which is only used in "none" break mode when
                    # add_eos_for_other_targets is False
                    past_target = torch.cat([past_target.new([self.vocab.pad()]), past_target[1:], source[-2, None]])

            for t in self.targets:
                if t == 'self':
                    target.append(source)
                elif t == 'future':
                    target.append(future_target)
                elif t == 'past':
                    target.append(past_target)
                else:
                    raise Exception('invalid target ' + t)

            if len(target) == 1:
                target = target[0]
        else:
            target = future_target

        return source, self._filter_vocab(target)

    def _maybe_add_bos(self, source, target):
        if self.add_bos_token:
            source = torch.cat([source.new([self.vocab.bos()]), source])
            if target is not None:
                target = torch.cat([target.new([self.tgt_vocab.bos()]), target])
        return source, target

    def _filter_vocab(self, target):
        if len(self.tgt_vocab) != len(self.vocab):
            def _filter(target):
                mask = target.ge(len(self.tgt_vocab))
                if mask.any():
                    target[mask] = self.tgt_vocab.unk()
                return target

            if isinstance(target, list):
                return [_filter(t) for t in target]
            return _filter(target)
        return target

    def _maybe_pad(self, source, target):
        if self.pad_to_a_length > 0:
            source = torch.cat([source, source.new([self.vocab.pad()] * (self.pad_to_a_length - len(source)))])
            if target is not None:
                target = torch.cat([target, target.new([self.tgt_vocab.pad()] * (self.pad_to_a_length - len(target)))])
        return source, target

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        return collate(samples, self.vocab.pad(), self.vocab.eos())

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        if hasattr(self.dataset, 'max_example_size') and self.pad_to_a_length > 0:
            return self.dataset.max_example_size
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if hasattr(self.dataset, 'max_example_size') and self.pad_to_a_length > 0:
            return self.dataset.max_example_size
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    def set_epoch(self, epoch):
        logger.info('Monolingual dataset reindex at the beginning of epoch {}!'.format(epoch))
        # assert isinstance(self.dataset, TokenBlockDataset)
        self.dataset.reindex(epoch)
        self.sizes = np.array(self.dataset.sizes)
