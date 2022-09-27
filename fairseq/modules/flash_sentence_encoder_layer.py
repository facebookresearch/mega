# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple, List, Union
import math

import torch
import torch.nn as nn

from fairseq.modules import (
    LayerNorm,
    ScaleNorm,
    GatedAttentionUnit,
)
from fairseq.modules.fairseq_dropout import FairseqDropout


class FlashSentenceEncoderLayer(nn.Module):
    """
        Implements a Flash-Quad encoder layer.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        z_dim: int = 128,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        norm_type: str = 'scalenorm',
        max_positions: int = 1024,
        export: bool = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)

        self.gau = self.build_gated_attention_unit(embedding_dim, hidden_dim, z_dim, attention_dropout, hidden_dropout, max_positions)
        self.pre_norm = self.build_norm_layer(norm_type, embedding_dim, export)

    def build_norm_layer(self, norm_type, embedding_dim, export):
        if norm_type == 'layernorm':
            return LayerNorm(embedding_dim, export=export)
        elif norm_type == 'scalenorm':
            return ScaleNorm(dim=-1)
        elif norm_type == 'batchnorm':
            return nn.BatchNorm1d(embedding_dim)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

    def build_gated_attention_unit(self, embedding_dim, hidden_dim, z_dim, attention_dropout, hidden_dropout, max_positions):
        return GatedAttentionUnit(
            embed_dim=embedding_dim,
            zdim=z_dim,
            hdim=hidden_dim,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_positions=max_positions
        )

    def normalize(self, x):
        if isinstance(self.pre_norm, nn.BatchNorm1d):
            assert x.dim() == 3
            x = x.permute(1, 2, 0)
            x = self.pre_norm(x)
            return x.permute(2, 0, 1)
        else:
            return self.pre_norm(x)

    def forward(
        self,
        x: torch.Tensor,
        x_padding_mask: Optional[torch.Tensor] = None,
    ):
        residual = x
        x = self.normalize(x)

        x, attn = self.gau(x, x_padding_mask)

        x = self.dropout_module(x)
        x = residual + x

        return x, attn
