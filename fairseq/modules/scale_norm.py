# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.scalar = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('scalar', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.scalar, 1.0)

    def forward(self, x):
        mean_square = torch.mean(torch.square(x), dim=self.dim, keepdim=True)
        if self.scalar is not None:
            x = self.scalar * x

        x = x * torch.rsqrt(mean_square + self.eps)
        return x

    def extra_repr(self) -> str:
        return 'dim={dim}, eps={eps}, affine={affine}'.format(**self.__dict__)
