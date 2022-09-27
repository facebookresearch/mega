# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn

from fairseq.modules import FairseqDropout, RealNumberEmbedding
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_


class LSTMLRAEncoder(nn.Module):
    """
    Implementation for a Bi-directional LSTM based Sentence Encoder.

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
        num_layers: int = 6,
        bidirectional=False,
        embedding_type: str = "sparse",
        embedding_dim: int = 768,
        hidden_dim: int = 3072,
        output_dropout: float = 0.0,
        input_dropout: float = 0.0,
        max_seq_len: int = 256,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        sen_rep_type: str = 'cls'
    ) -> None:

        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.sen_rep_type = sen_rep_type
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_in_module = FairseqDropout(input_dropout, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(output_dropout, module_name=self.__class__.__name__)
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU

        assert embedding_type in ['sparse', 'linear']
        self.embed_tokens = self.build_embedding(self.embedding_type, self.vocab_size, self.embedding_dim, self.padding_idx)

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.lstm = LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=self.dropout_out_module.p if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )

    def build_embedding(self, embedding_type, vocab_size, embedding_dim, padding_idx):
        if embedding_type == 'sparse':
            embed_tokens = nn.Embedding(vocab_size, embedding_dim, padding_idx)
            nn.init.normal_(embed_tokens.weight, mean=0, std=embedding_dim ** -0.5)
            return embed_tokens
        else:
            embed_tokens = RealNumberEmbedding(embedding_dim)
            return embed_tokens

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(
            self,
            tokens: torch.Tensor,
            src_lengths: torch.Tensor,
            enforce_sorted: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        bsz, seqlen = tokens.size()
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

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if padding_mask is not None:
            # pack embedded source tokens into a PackedSequence
            packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data, enforce_sorted=enforce_sorted)
        else:
            packed_x = x

        x, (h, c) = self.lstm(packed_x)

        if padding_mask is not None:
            # unpack outputs and apply dropout
            x, _ = nn.utils.rnn.pad_packed_sequence(x, padding_value=self.padding_idx * 1.0)

        if self.sen_rep_type == 'mp':
            sentence_rep = x.sum(dim=0) / src_lengths.unsqueeze(1)
        else:
            sentence_rep = h[-2:] if self.bidirectional else h[-1:]
            sentence_rep = sentence_rep.transpose(0, 1).reshape(bsz, -1)

        return x, sentence_rep


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        if 'bias' in name:
            nn.init.constant_(param, 0.)
    return m
