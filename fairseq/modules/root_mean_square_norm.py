# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, number_features, eps=1e-6, affine=True):
        super().__init__()
        self.num_features = number_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.weight, 1.0)

    def forward(self, x):
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        if self.weight is not None:
            x = x * self.weight

        x = x * torch.rsqrt(mean_square + self.eps)
        return x

    def extra_repr(self) -> str:
        return '{num_features}, eps={eps}, affine={affine}'.format(**self.__dict__)
