# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# for speech command task

import logging
import os

from fairseq.data import Dictionary, SpeechCommandsDataset
from fairseq.tasks import FairseqTask, register_task

logger = logging.getLogger(__name__)

@register_task('speech_commands')
class SpeechCommandsTask(FairseqTask):
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

        # speech commands arguments
        parser.add_argument('--sc-all-classes', action='store_true', default=False)
        parser.add_argument('--sc-dropped-rate', type=float, default=0.0)
        parser.add_argument('--mfcc', action='store_true', default=False)

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

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        raise NotImplementedError

    @classmethod
    def setup_task(cls, args, **kwargs):
        return SpeechCommandsTask(args)

    def load_dataset(self, split, combine=False, **kwargs):
        """gtjjjLoad a given dataset split (e.g., train, valid, test)."""
        # def get_path(type, split):
        #     return os.path.join(self.args.data, type, split)

        dataset = SpeechCommandsDataset(
            partition=split,
            length=16000,  # self.L,
            mfcc=self.args.mfcc,
            sr=1,
            dropped_rate=self.args.sc_dropped_rate,
            path=self.args.data,
            all_classes=self.args.sc_all_classes,
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
