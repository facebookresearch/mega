# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os

import numpy as np
import torch

from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    data_utils,
    Dictionary,
    IdDataset,
    FairseqDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TransformEosDataset,
    TruncatedDictionary,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task("language_modeling")
class LanguageModelingTask(FairseqTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--sample-break-mode', default='none',
                            choices=['none', 'complete', 'complete_doc', 'complete_doc_with_boundary', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--output-dictionary-size', default=-1, type=int,
                            help='limit the size of output dictionary')
        parser.add_argument('--self-target', action='store_true',
                            help='include self target')
        parser.add_argument('--future-target', action='store_true',
                            help='include future target')
        parser.add_argument('--past-target', action='store_true',
                            help='include past target')
        parser.add_argument('--add-bos-token', action='store_true',
                            help='prepend beginning of sentence token (<s>)')
        parser.add_argument('--max-target-positions', type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')
        parser.add_argument('--variant-block-multiple-min', default=1, type=int,
                            help='the block size is at least this multiple of one chunk size')
        parser.add_argument('--variant-block-multiple-max', default=1, type=int,
                            help='the block size is at most this multiple of one chunk size')
        parser.add_argument('--valid-block', default="size:100000000", help="either size:value or splits:value, "
                                                                            "the former is the block size, "
                                                                            "the latter is the number of blocks")
        # fmt: on

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary or dictionary
        # added by Chunting: if chunk_size > 0, this is for Mega LM which takes K chunks as a training example
        # and one chunk as a valid/test example
        self.chunk_size = args.decoder_chunk_size if hasattr(args, 'decoder_chunk_size') else -1
        self.is_mega_lm = True if hasattr(args, 'decoder_chunk_size') else False
        if targets is None:
            targets = ["future"]
        self.targets = targets

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = None
        output_dictionary = None
        if args.data:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
            logger.info("dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    dictionary, args.output_dictionary_size
                )
        return (dictionary, output_dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary, output_dictionary = cls.setup_dictionary(args, **kwargs)

        # upgrade old checkpoints
        if hasattr(args, "exclude_self_target"):
            args.self_target = not args.exclude_self_target

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        return cls(args, dictionary, output_dictionary, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)

        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    "Unsupported language modeling target: {}".format(target)
                )

        return model

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )

        if split == 'train':
            chunk_size = self.args.decoder_chunk_size if self.is_mega_lm else self.args.tokens_per_sample
        else:
            if self.is_mega_lm:
                # at inference, read data by documents for mega lm by setting chunk_size to be a very large number
                k, v = self.args.valid_block.split(":")
                chunk_size = int(v) if k == "size" else math.ceil(sum(dataset.sizes) / float(v))
                # to prevent the samples being filtered
                if k == "splits":
                    # when k is splits, one sample can be extremely long as chunk_size,
                    self.args.max_tokens_valid = chunk_size * 10
            else:
                chunk_size = self.args.tokens_per_sample
            self.chunk_size = chunk_size

        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            chunk_size,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
            variant_block_size_multiples=(self.args.variant_block_multiple_min,
                                          self.args.variant_block_multiple_max) if split == 'train' else (1, 1),
            seed=self.args.seed,
        )

        if split != 'train' and self.is_mega_lm:
            k, v = self.args.valid_block.split(":")
            # to prevent the samples being filtered
            if k == 'size':
                logger.info("Max document length of {} set = {}, number of documents = {}.".format(split, max(dataset.sizes), len(dataset.sizes)))
                self.args.max_tokens_valid = max(dataset.sizes) * 5

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )

        self.datasets[split] = self._initialize_dataset(
            dataset=dataset,
            sizes=dataset.sizes,
            src_vocab=self.dictionary,
            tgt_vocab=self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
        )

    def _initialize_dataset(self, **kwargs):
        return MonolingualDataset(**kwargs)

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.source_dictionary.bos()
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(
            dataset,
            token=self.source_dictionary.pad()
        )
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(src_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False),
            },
            sizes=[np.array(src_lengths)],
        )

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token,
            )

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary

    def valid_step(self, sample, model, criterion, incremental_states=None):
        if not self.is_mega_lm:
            loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
            return loss, sample_size, logging_output
        else:
            # todo: criterion currently limited to adaptive_loss
            assert incremental_states is not None
            model.eval()
            with torch.no_grad():
                loss, sample_size, logging_output = criterion(model, sample, incremental_states=incremental_states)
            return loss, sample_size, logging_output

    # we need to override get_batch_iterator because we want to reset the epoch iterator each time
    def get_batch_iterator(
            self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
            ignore_invalid_inputs=False, required_batch_size_multiple=1,
            seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=1,
            sharding=True,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert isinstance(dataset, FairseqDataset)
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]
        # dataset.dataset is token_block_dataset
        if (
            not hasattr(dataset.dataset, 'variant_block_multiple_max') or
                (hasattr(dataset.dataset, 'variant_block_multiple_max') and dataset.dataset.variant_block_multiple_max == 1)
        ):
            # valid / test of mega LM or normal LM
            if sharding:
                # normal LM
                batch_iter = super().get_batch_iterator(
                    dataset, max_tokens=max_tokens, max_sentences=max_sentences, max_positions=max_positions,
                    ignore_invalid_inputs=ignore_invalid_inputs, required_batch_size_multiple=required_batch_size_multiple,
                    seed=seed, num_shards=num_shards, shard_id=shard_id, num_workers=num_workers, epoch=epoch,
                )
            else:
                # Mega LM
                batch_iter = super().get_batch_iterator(
                    dataset, max_tokens=self.args.max_tokens_valid, max_sentences=max_sentences, max_positions=10000000,
                    ignore_invalid_inputs=ignore_invalid_inputs,
                    required_batch_size_multiple=required_batch_size_multiple,
                    seed=seed, num_shards=1, shard_id=0, num_workers=num_workers, epoch=epoch,
                )
            # already done in super().get_batch_iterator
            # self.dataset_to_epoch_iter[dataset] = batch_iter
            return batch_iter

        self.dataset_to_epoch_iter = {}
        epoch_iter = super().get_batch_iterator(
            dataset, max_tokens, max_sentences, max_positions,
            ignore_invalid_inputs, required_batch_size_multiple,
            seed, num_shards, shard_id, num_workers, epoch,
        )
        self.dataset_to_epoch_iter = {}
        return epoch_iter
