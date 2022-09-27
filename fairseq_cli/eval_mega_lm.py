#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os
from typing import Dict, List, Optional

import torch
from torch import Tensor

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import LMContextWindowDataset
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq import distributed_utils
import sys

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('fairseq_cli.eval_lm')


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """ increments counters for the sum of log probs of current word and next
            word (given context ending at current word). Since the next word might be at the end of the example,
            or it might be not counted because it is not an ending subword unit,
            also keeps track of how many of those we have seen """
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}'.format(self.word, self.count, self.log_prob, self.is_bpe,
                                               self.next_word_prob, self.count - self.missing_next_words)


def main(parsed_args, **unused_kwargs):
    assert parsed_args.path is not None, '--path required for evaluation!'

    if torch.cuda.is_available() and not parsed_args.cpu:
        torch.cuda.set_device(parsed_args.device_id)

    utils.import_user_module(parsed_args)

    logger.info(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    parsed_args.decoder_chunk_size = parsed_args.test_chunk_size # a hack
    task = tasks.setup_task(parsed_args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(parsed_args.path))
    models, args = checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides=eval(parsed_args.model_overrides),
        task=task,
        suffix=getattr(parsed_args, "checkpoint_suffix", ""),
    )

    for arg in vars(parsed_args).keys():
        if arg not in {
            'self_target', 'future_target', 'past_target', 'tokens_per_sample',
            'output_size_dictionary', 'add_bos_token',
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    # reduce tokens per sample by the required context window size
    args.tokens_per_sample -= args.context_window

    logger.info("Loaded args!")
    logger.info(args)

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
        output_file = open(output_path, 'w', buffering=1, encoding='utf-8')
    else:
        output_file = sys.stdout

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    dataset = task.dataset(args.gen_subset)
    if args.context_window > 0:
        dataset = LMContextWindowDataset(
            dataset=dataset,
            tokens_per_sample=args.tokens_per_sample,
            context_window=args.context_window,
            pad_idx=task.source_dictionary.pad(),
        )
    print('{} {} {} examples'.format(args.data, args.gen_subset, len(dataset)), file=output_file)

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.prepare_for_inference_(args)
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    assert len(models) > 0

    logger.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))
    task = tasks.setup_task(args)

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=max(250000, task.chunk_size),
        max_sentences=args.max_sentences,
        max_positions=1000000,
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        sharding=False,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, args.softmax_batch)

    score_sum = 0.
    count = 0

    if args.remove_bpe is not None:
        if args.remove_bpe == 'sentencepiece':
            raise NotImplementedError
        else:
            bpe_cont = args.remove_bpe.rstrip()
            bpe_toks = {
                i
                for i in range(len(task.source_dictionary))
                if task.source_dictionary[i].endswith(bpe_cont)
            }
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()

    wps_meter = TimeMeter()

    chunk_size = args.decoder_chunk_size * args.chunk_nums
    for doc_sample in progress:
        if 'net_input' not in doc_sample:
            continue

        # a specific assertion for debugging
        total_size = doc_sample['net_input']['src_tokens'].size(1)
        batch_size = len(doc_sample['net_input']['src_lengths'])
        incremental_states = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})

        for kk in range(0, total_size, chunk_size):
            sample = {
                'id': doc_sample['id'],
                'nsentences': doc_sample['nsentences'],
                'ntokens': doc_sample['target'][:, kk: kk + chunk_size].ne(task.dictionary.pad()).sum().item(),
                'net_input': {
                    'src_tokens': doc_sample['net_input']['src_tokens'][:, kk: kk + chunk_size],
                    'src_lengths': doc_sample['net_input']['src_tokens'].ne(task.dictionary.pad()).sum(1),
                },
                'target': doc_sample['target'][:, kk: kk + chunk_size],
            }

            sample = utils.move_to_cuda(sample) if use_cuda else sample

            gen_timer.start()
            hypos = scorer.generate(models, sample, incremental_states=incremental_states)
            gen_timer.stop(sample['ntokens'])

            for j, hypos_i in enumerate(hypos):
                hypo = hypos_i[0]
                sample_id = sample['id'][j]

                tokens = hypo['tokens']
                tgt_len = tokens.numel()
                pos_scores = hypo['positional_scores'].float()

                if args.add_bos_token:
                    assert hypo['tokens'][0].item() == task.target_dictionary.bos()
                    tokens = tokens[1:]
                    pos_scores = pos_scores[1:]

                skipped_toks = 0
                if bpe_toks is not None:
                    for i in range(tgt_len - 1):
                        if tokens[i].item() in bpe_toks:
                            skipped_toks += 1
                            pos_scores[i + 1] += pos_scores[i]
                            pos_scores[i] = 0

                inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                if inf_scores.any():
                    logger.info(
                        'skipping tokens with inf scores:',
                        task.target_dictionary.string(tokens[inf_scores.nonzero()])
                    )
                    pos_scores = pos_scores[(~inf_scores).nonzero()]
                score_sum += pos_scores.sum().cpu()
                count += pos_scores.numel() - skipped_toks

                if args.output_word_probs or args.output_word_stats:
                    w = ''
                    word_prob = []
                    is_bpe = False
                    for i in range(len(tokens)):
                        w_ind = tokens[i].item()
                        w += task.source_dictionary[w_ind]
                        if bpe_toks is not None and w_ind in bpe_toks:
                            w = w[:-bpe_len]
                            is_bpe = True
                        else:
                            word_prob.append((w, pos_scores[i].item()))

                            next_prob = None
                            ind = i + 1
                            while ind < len(tokens):
                                if pos_scores[ind].item() != 0:
                                    next_prob = pos_scores[ind]
                                    break
                                ind += 1

                            word_stats.setdefault(w, WordStat(w, is_bpe)).add(pos_scores[i].item(), next_prob)
                            is_bpe = False
                            w = ''
                    if args.output_word_probs:
                        print(
                            'S-' + str(int(sample_id)) + " "
                            + ('\t'.join('{} [{:2f}]'.format(x[0], x[1]) for x in word_prob)), file=output_file
                        )

            wps_meter.update(sample['ntokens'])
            progress.log({'wps': round(wps_meter.avg)})

    avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
    print('Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(
        count, gen_timer.sum, 1. / gen_timer.avg, file=output_file
    ))
    print('Loss (base 2): {:.4f}, Perplexity: {:.2f}'.format(
        avg_nll_loss, 2**avg_nll_loss, file=output_file
    ))

    if args.output_word_stats:
        for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
            logger.info(ws)

    if args.results_path is not None:
        output_file.close()

def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)


if __name__ == '__main__':
    cli_main()
