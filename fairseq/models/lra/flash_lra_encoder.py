# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, List, Union
import math

import torch
import torch.nn as nn

from fairseq.modules import (
    LayerDropModuleList,
    RealNumberEmbedding,
    FlashSentenceEncoderLayer,
)
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.scale_norm import ScaleNorm
from fairseq.modules.layer_norm import LayerNorm


class FlashLRAEncoder(nn.Module):
    """
    Implementation for a Bi-directional FLASH based Sentence Encoder used
    in masked pre-trained language models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_type: str = "sparse",
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        z_dim: int = 128,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        norm_type: str = 'scalenorm',
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        export: bool = False,
        traceable: bool = False,
        sen_rep_type: str = 'cls',
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.sen_rep_type = sen_rep_type

        assert embedding_type in ['sparse', 'linear']
        self.embed_tokens = self.build_embedding(self.embedding_type, self.vocab_size, self.embedding_dim, self.padding_idx)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers

        self.layers.extend([
            self.build_flash_sentence_encoder_layer(
                embedding_dim=self.embedding_dim,
                hidden_dim=hidden_dim,
                z_dim=z_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                norm_type=norm_type,
                max_positions=self.max_seq_len,
                export=export
            )
            for _ in range(self.num_layers)
        ])

        self.final_norm = self.build_final_norm_layer(norm_type, embedding_dim, export)

    def build_final_norm_layer(self, norm_type, embedding_dim, export):
        if norm_type == 'layernorm':
            return LayerNorm(embedding_dim, export=export)
        elif norm_type == 'scalenorm':
            return ScaleNorm(dim=-1)
        elif norm_type == 'batchnorm':
            return nn.BatchNorm1d(embedding_dim)
        else:
            raise ValueError('Unknown norm type: {}'.format(norm_type))

    def build_embedding(self, embedding_type, vocab_size, embedding_dim, padding_idx):
        if embedding_type == 'sparse':
            embed_tokens = nn.Embedding(vocab_size, embedding_dim, padding_idx)
            nn.init.normal_(embed_tokens.weight, mean=0, std=embedding_dim ** -0.5)
            return embed_tokens
        else:
            embed_tokens = RealNumberEmbedding(embedding_dim)
            return embed_tokens

    def build_flash_sentence_encoder_layer(
        self,
        embedding_dim,
        hidden_dim,
        z_dim,
        dropout,
        attention_dropout,
        hidden_dropout,
        norm_type,
        max_positions,
        export,
    ):
        return FlashSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            norm_type=norm_type,
            max_positions=max_positions,
            export=export
        )

    def normalize(self, x):
        if isinstance(self.final_norm, nn.BatchNorm1d):
            assert x.dim() == 3
            x = x.permute(1, 2, 0)
            x = self.final_norm(x)
            return x.permute(2, 0, 1)
        else:
            return self.final_norm(x)

    def forward(
            self,
            tokens: torch.Tensor,
            src_lengths: torch.Tensor,
            last_state_only: bool = False,
            positions: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        if self.embedding_type == 'sparse':
            padding_mask = tokens.eq(self.padding_idx)
            if not self.traceable and not self.tpu and not padding_mask.any():
                padding_mask = None
            # B x T -> B x T x D
            x = self.embed_tokens(tokens)
        else:
            padding_mask = None
            # B x T -> B x T x 1 -> B x T x D
            x = self.embed_tokens(tokens)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for i in range(self.num_layers):
            x, _ = self.layers[i](x, x_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        x = self.normalize(x)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.transpose(0, 1).unsqueeze(-1), 0.0)

        if self.sen_rep_type == 'mp':
            sentence_rep = x.sum(dim=0) / src_lengths.unsqueeze(1)
        else:
            sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep
