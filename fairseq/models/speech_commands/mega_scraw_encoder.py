# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn

from fairseq.modules import (
    SequenceNorm,
    RealNumberEmbedding,
    LayerDropModuleList,
    MegaSentenceEncoderLayer,
)
from fairseq.modules.fairseq_dropout import FairseqDropout


class MegaSCRawEncoder(nn.Module):
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
        num_encoder_layers: int = 6,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        ffn_hidden_dim: int = 1024,
        z_dim: int = 128,
        n_dim: int = 16,
        activation: str = 'silu',
        attention_activation: str = 'softmax',
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        chunk_size: int = -1,
        norm_type: str = 'layernorm',
        normalize_before: bool = False,
        feature_dropout: bool = False,
        layerdrop: float = 0.0,
        truncation: int = None,
        rel_pos_bias: str = 'simple',
        max_seq_len: int = 16000,
        export: bool = False,
        traceable: bool = False,
        sen_rep_type: str = 'cls',
    ) -> None:

        super().__init__()
        self.embedding_dropout = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.chunk_size = chunk_size
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.sen_rep_type = sen_rep_type

        self.embed_tokens = RealNumberEmbedding(embedding_dim)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers

        self.layers.extend([
            self.build_mega_sentence_encoder_layer(
                embedding_dim=self.embedding_dim,
                hidden_dim=hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                z_dim=z_dim,
                n_dim=n_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                chunk_size=chunk_size,
                truncation=truncation,
                rel_pos_bias=rel_pos_bias,
                max_positions=self.max_seq_len,
                activation=activation,
                attention_activation=attention_activation,
                norm_type=norm_type,
                prenorm=normalize_before,
                feature_dropout=feature_dropout,
                export=export
            )
            for _ in range(self.num_layers)
        ])

        if normalize_before:
            self.final_norm = SequenceNorm(norm_type, embedding_dim, export=export)
        else:
            self.final_norm = None

    def build_mega_sentence_encoder_layer(
        self,
        embedding_dim,
        hidden_dim,
        ffn_hidden_dim,
        z_dim,
        n_dim,
        dropout,
        attention_dropout,
        hidden_dropout,
        chunk_size,
        truncation,
        rel_pos_bias,
        max_positions,
        activation,
        attention_activation,
        norm_type,
        prenorm,
        feature_dropout,
        export,
    ):
        return MegaSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            z_dim=z_dim,
            n_dim=n_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            chunk_size=chunk_size,
            truncation=truncation,
            rel_pos_bias=rel_pos_bias,
            max_positions=max_positions,
            activation=activation,
            attention_activation=attention_activation,
            norm_type=norm_type,
            prenorm=prenorm,
            feature_dropout=feature_dropout,
            export=export
        )

    def forward(
            self,
            tokens: torch.Tensor,
            src_lengths: torch.Tensor,
            last_state_only: bool = False,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:

        bsz, seq_len = tokens.size()
        assert self.chunk_size <= 0 or seq_len % self.chunk_size == 0, 'sequence length {} must be divided by chunk size {}'.format(seq_len, self.chunk_size)

        padding_mask = None
        # B x T -> B x T x D
        x = self.embed_tokens(tokens)
        x = self.embedding_dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for i in range(self.num_layers):
            x, _ = self.layers[i](x, x_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

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
