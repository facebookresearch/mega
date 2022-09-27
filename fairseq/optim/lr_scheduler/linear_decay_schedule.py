# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('linear_decay')
class LinearDecaySchedule(FairseqLRScheduler):
    """Decay the LR on a linear schedule.
    """

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with step.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = args.lr[0]
        if args.warmup_updates < 0:
            raise ValueError('warm up steps cannot be negative.')
        elif args.warmup_updates == 0:
            assert args.warmup_init_lr < 0
            args.warmup_init_lr = warmup_end_lr
        else:
            assert args.warmup_init_lr < warmup_end_lr
            if args.warmup_init_lr < 0:
                args.warmup_init_lr = 0

        # linearly warmup for the first args.warmup_updates
        if args.warmup_updates > 0:
            self.warmup_power = args.warmup_power
            self.warmup_factor = (warmup_end_lr - args.warmup_init_lr) / (args.warmup_updates ** args.warmup_power)
        else:
            self.warmup_power = 1
            self.warmup_factor = 0

        self.end_learning_rate = args.end_learning_rate
        self.total_num_update = args.total_num_update
        self.lr_factor = (warmup_end_lr - self.end_learning_rate) / (self.total_num_update - args.warmup_updates)

        # initial learning rate
        self.lr = args.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-power', default=1, type=int, metavar='N', help='the power of warmup')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        parser.add_argument('--end-learning-rate', default=0.0, type=float)
        parser.add_argument('--total-num-update', default=1000000, type=int)

    def state_dict(self):
        return {'lr': self.lr}

    def load_state_dict(self, state_dict):
        if 'lr' in state_dict:
            self.lr = state_dict['lr']

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates <= self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + (num_updates ** self.warmup_power) * self.warmup_factor
        elif num_updates >= self.total_num_update:
            self.lr = self.end_learning_rate
        else:
            self.lr = self.lr_factor * (self.total_num_update - num_updates) + self.end_learning_rate

        self.optimizer.set_lr(self.lr)
        return self.lr
