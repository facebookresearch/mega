# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import math

import torch
import torch.nn as nn

from fairseq.modules.moving_average_gated_attention import MovingAverageGatedAttention
from fairseq.modules.normalized_feedforward_network import NormalizedFeedForwardNetwork


class MegaSentenceEncoderLayer(nn.Module):
    """
        Implements a Flash-Quad encoder layer.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        ffn_hidden_dim: int = 1024,
        z_dim: int = 128,
        n_dim: int = 16,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        chunk_size: int = -1,
        truncation: int = None,
        rel_pos_bias = 'simple',
        max_positions: int = 1024,
        activation='silu',
        attention_activation: str = 'softmax',
        norm_type: str = 'layernorm',
        prenorm: bool = True,
        feature_dropout: bool = False,
        export: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.mega_layer = self.build_mega_layer(embedding_dim, hidden_dim, z_dim, n_dim,
                                           dropout, attention_dropout, hidden_dropout,
                                           activation, attention_activation,
                                           chunk_size, truncation, rel_pos_bias, max_positions,
                                           norm_type, prenorm, feature_dropout, export)

        if ffn_hidden_dim is not None and ffn_hidden_dim > 0:
            self.nffn = self.build_nffn_layer(embedding_dim, ffn_hidden_dim,
                                              dropout, hidden_dropout, activation,
                                              norm_type, prenorm, feature_dropout, export)
        else:
            self.nffn = None

    def build_mega_layer(self, embedding_dim, hidden_dim, z_dim, n_dim,
                         dropout, attention_dropout, hidden_dropout,
                         activation, attention_activation,
                         chunk_size, truncation, rel_pos_bias, max_positions,
                         norm_type, prenorm, feature_dropout, export):
        return MovingAverageGatedAttention(
            embed_dim=embedding_dim,
            zdim=z_dim,
            hdim=hidden_dim,
            ndim=n_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            chunk_size=chunk_size,
            truncation=truncation,
            rel_pos_bias=rel_pos_bias,
            max_positions=max_positions,
            activation=activation,
            attention_activation=attention_activation,
            bidirectional=True,
            norm_type=norm_type,
            prenorm=prenorm,
            feature_dropout=feature_dropout,
            export=export
        )

    def build_nffn_layer(self, embedding_dim, ffn_hidden_dim,
                         dropout, hidden_dropout, activation,
                         norm_type, prenorm, feature_dropout, export):
        return NormalizedFeedForwardNetwork(
            embed_dim=embedding_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            hidden_dropout=hidden_dropout,
            activation=activation,
            norm_type=norm_type,
            prenorm=prenorm,
            feature_dropout=feature_dropout,
            export=export
        )

    def forward(
        self,
        x: torch.Tensor,
        x_padding_mask: Optional[torch.Tensor] = None,
    ):

        seq_len = x.size(0)
        if self.chunk_size > 0:
            assert seq_len % self.chunk_size == 0, 'the input sequence length {} cannot be divided by chunk size {}'.format(seq_len, self.chunk_size)
        x, attn = self.mega_layer(x, x_padding_mask)

        if self.nffn is not None:
            x = self.nffn(x)

        return x, attn
