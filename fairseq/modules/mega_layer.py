# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq.modules.moving_average_gated_attention import MovingAverageGatedAttention
from fairseq.modules.gated_cross_attention import GatedCrossAttention
from fairseq.modules.normalized_feedforward_network import NormalizedFeedForwardNetwork
from torch import Tensor


class MegaEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.mega_layer = self.build_mega_layer(self.embed_dim, args)
        if args.encoder_ffn_embed_dim > 0:
            self.nffn = self.build_nffn_layer(self.embed_dim, args)
        else:
            self.nffn = None

    def build_mega_layer(self, embed_dim, args):
        return MovingAverageGatedAttention(
            embed_dim=embed_dim,
            zdim=args.encoder_z_dim,
            hdim=args.encoder_hidden_dim,
            ndim=args.encoder_n_dim,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            hidden_dropout=args.hidden_dropout,
            chunk_size=args.encoder_chunk_size,
            truncation=args.truncation_length,
            rel_pos_bias=args.rel_pos_bias,
            max_positions=args.max_source_positions,
            activation=args.activation_fn,
            attention_activation=args.attention_activation_fn,
            bidirectional=True,
            norm_type=args.normalization_type,
            prenorm=args.normalize_before,
            feature_dropout=args.feature_dropout,
        )

    def build_nffn_layer(self, embed_dim, args):
        return NormalizedFeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_hidden_dim=args.encoder_ffn_embed_dim,
            dropout=args.dropout,
            hidden_dropout=args.activation_dropout,
            activation=args.activation_fn,
            norm_type=args.normalization_type,
            prenorm=args.normalize_before,
            feature_dropout=args.feature_dropout,
        )

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        x, _ = self.mega_layer(x, encoder_padding_mask)
        if self.nffn is not None:
            x = self.nffn(x)

        return x


class MegaDecoderLayer(nn.Module):
    """Decoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, no_cross_attention=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.mega_layer = self.build_mega_layer(self.embed_dim, args)
        self.cross_attn = None if no_cross_attention else self.build_cross_attn(self.embed_dim, args)
        if args.decoder_ffn_embed_dim > 0:
            self.nffn = self.build_nffn_layer(self.embed_dim, args)
        else:
            self.nffn = None

        self.need_attn = False
        self.onnx_trace = False

    def build_mega_layer(self, embed_dim, args):
        return MovingAverageGatedAttention(
            embed_dim=embed_dim,
            zdim=args.decoder_z_dim,
            hdim=args.decoder_hidden_dim,
            ndim=args.decoder_n_dim,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            hidden_dropout=args.hidden_dropout,
            chunk_size=args.decoder_chunk_size,
            truncation=args.truncation_length,
            rel_pos_bias=args.rel_pos_bias,
            max_positions=args.max_target_positions,
            activation=args.activation_fn,
            attention_activation=args.attention_activation_fn,
            bidirectional=False,
            norm_type=args.normalization_type,
            prenorm=args.normalize_before,
            feature_dropout=args.feature_dropout,
        )

    def build_cross_attn(self, embed_dim, args):
        return GatedCrossAttention(
            embed_dim=embed_dim,
            zdim=args.decoder_z_dim,
            ndim=args.decoder_n_dim,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            hidden_dropout=args.hidden_dropout,
            activation=args.activation_fn,
            attention_activation=args.attention_activation_fn,
            norm_type=args.normalization_type,
            prenorm=args.normalize_before,
            feature_dropout=args.feature_dropout,
            rel_pos_bias=args.rel_pos_bias,
            max_positions=max(args.max_target_positions, args.max_source_positions),
        )

    def build_nffn_layer(self, embed_dim, args):
        return NormalizedFeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_hidden_dim=args.decoder_ffn_embed_dim,
            dropout=args.dropout,
            hidden_dropout=args.activation_dropout,
            activation=args.activation_fn,
            norm_type=args.normalization_type,
            prenorm=args.normalize_before,
            feature_dropout=args.feature_dropout,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        decoder_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_out (Tensor): encoder out for cross attention `(src_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary ByteTensor of shape `(batch, src_len)` where padding elements are indicated by ``1``.
            incremental_state: dictionary for caching incremental states.
            attn_mask (Tensor): attention mask for autoregressive decoding.
            decoder_padding_mask: padding mask for target sequence.
            need_attn (bool, optional): return attention weights.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        x, attn = self.mega_layer(x=x, padding_mask=decoder_padding_mask,
                                  incremental_state=incremental_state,
                                  need_weights=False, attn_mask=attn_mask)

        if self.cross_attn is not None:
            x, attn = self.cross_attn(query=x, key=encoder_out, value=encoder_out,
                                      padding_mask=decoder_padding_mask,
                                      key_padding_mask=encoder_padding_mask,
                                      incremental_state=incremental_state,
                                      static_kv=True, need_weights=need_attn)

        if self.nffn is not None:
            x = self.nffn(x)

        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
