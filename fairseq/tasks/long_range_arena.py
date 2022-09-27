# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq import utils
from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PixelSequenceDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    TruncateDataset
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task('lra-text')
class LRATextTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes or regression targets')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')

    def __init__(self, args, data_dictionary, label_dictionary, cls_idx):
        super().__init__(args)
        self.cls_idx = cls_idx
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        self.prepend_cls = args.sen_rep_type == 'cls'
        if not hasattr(args, 'max_positions'):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):

        # load data dictionary
        data_dict = cls.load_dictionary(os.path.join(args.data, 'src-bin', 'dict.txt'))
        cls_idx = data_dict.add_symbol('<CLS>')
        logger.info('[input] dictionary: {} types'.format(len(data_dict)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(os.path.join(args.data, 'label-bin', 'dict.txt'))
            logger.info('[label] dictionary: {} types'.format(len(label_dict)))
        else:
            label_dict = data_dict
        return LRATextTask(args, data_dict, label_dict, cls_idx)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        src_ds = make_dataset('src-bin', self.source_dictionary)
        if self.prepend_cls:
            src_ds = PrependTokenDataset(src_ds, self.cls_idx)
        src1_ds = make_dataset('src1-bin', self.source_dictionary)
        if src1_ds is not None:
            if self.prepend_cls:
                src1_ds = PrependTokenDataset(src1_ds, self.cls_idx)
            src1_tokens = TruncateDataset(src1_ds, self.args.max_positions)
        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_ds))

        src_tokens = TruncateDataset(src_ds, self.args.max_positions)
        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }
        if src1_ds is not None:
            dataset.update(
                net_input1={
                    'src_tokens': RightPadDataset(
                        src1_tokens,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    'src_lengths': NumelDataset(src1_tokens, reduce=False),
                },
            )

        label_dataset = make_dataset('label-bin', self.label_dictionary)
        if label_dataset is not None:
            dataset.update(
                target=OffsetTokensDataset(
                    StripTokenDataset(
                        label_dataset,
                        id_to_strip=self.label_dictionary.eos(),
                    ),
                    offset=-self.label_dictionary.nspecial,
                )
            )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )
        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary


@register_task('lra-image')
class LRAImageTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes or regression targets')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')
        parser.add_argument('--pixel-normalization', type=float, nargs='+', default=None, help='mean and std for pixel normalization.')

    def __init__(self, args):
        super().__init__(args)
        if not hasattr(args, 'max_positions'):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions
        self.normalization = (0.5, 0.5) if args.pixel_normalization is None else args.pixel_normalization
        assert len(self.normalization) == 2

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        raise NotImplementedError

    @classmethod
    def setup_task(cls, args, **kwargs):
        return LRAImageTask(args)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type):
            split_path = get_path(type, split)
            dataset = PixelSequenceDataset(split_path + '.src', self.normalization)
            return dataset

        src_ds = make_dataset('input')
        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_ds))

        src_tokens = TruncateDataset(src_ds, self.args.max_positions)
        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }

        label_path = get_path('label', split) + '.label'
        if os.path.exists(label_path):
            label_dataset = RawLabelDataset([int(line.strip()) for i, line in enumerate(open(label_path).readlines())])
            dataset.update(target=label_dataset)

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )
        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        return model

    def max_positions(self):
        return self._max_positions

    @property
    def target_dictionary(self):
        return None
