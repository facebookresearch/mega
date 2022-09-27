# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqLanguageModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    SequenceNorm,
    AdaptiveInput,
    MegaDecoderLayer
)
from torch import Tensor
import logging
logger = logging.getLogger(__name__)

DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("mega_lm")
class MegaLanguageModel(FairseqLanguageModel):
    """
    Args:
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, decoder):
        super().__init__(decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn', choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--attention-activation-fn', choices=['softmax', 'relu2', 'laplace'],
                            help='activation function for attention mechanism')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--hidden-dropout', type=float, metavar='D',
                            help='dropout probability for hidden vectors in Mega.')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--feature-dropout', action='store_true',
                            help='apply feature dropout')

        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-hidden-dim', type=int, metavar='N',
                            help='decoder hidden dimension for Mega')
        parser.add_argument('--decoder-z-dim', type=int, metavar='N',
                            help='decoder z dimension for Mega')
        parser.add_argument('--decoder-n-dim', type=int, metavar='N',
                            help='decoder n dimension for Mega')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-chunk-size', type=int, metavar='N',
                            help='chunk size of Mega decoder.')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')

        parser.add_argument('--decoder-input-dim', type=int, metavar='N',
                            help='decoder input dimension')

        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')

        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')

        parser.add_argument('--normalization-type', choices=['layernorm', 'scalenorm'],
                            help='normalization type')
        parser.add_argument('--normalize-before', action='store_true',
                            help='apply normalization layer before each encoder block')
        parser.add_argument('--no-affine-final-norm', action='store_true',
                            help='no affine parameters in the final norm.')
        parser.add_argument('--normalize-embedding', action='store_true',
                            help='normalize embedding for Mega.')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--rel-pos-bias', choices=['simple', 'rotary'], default='simple')
        parser.add_argument('--truncation-length', type=int, metavar='N',
                            help='truncation length of moving average layer.')

        # analysis of alpha
        parser.add_argument('--report-ema-alpha', action='store_true', default=False,
                            help='report the ema weights statistics at the end of each epoch')
        parser.add_argument('--write-out-alpha', action='store_true', default=False,
                            help='write the ema weights to files at the end of training')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict = task.source_dictionary

        if args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary), task.source_dictionary.pad(), args.decoder_input_dim,
                args.adaptive_input_factor, args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
            )
        else:
            embed_tokens = cls.build_embedding(args, task.source_dictionary, args.decoder_input_dim)

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)
            assert args.decoder_input_dim == args.decoder_embed_dim

        decoder = MegaDecoderNoCrossAttn(args, src_dict, embed_tokens)
        return cls(args, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = True,
        features_only: bool = False,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        decoder_out = self.decoder(
            src_tokens,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    def analyze_ema(self, report=False, output=False):
        # todo: fixme
        if self.args.report_ema_alpha and report:
            with torch.no_grad():
                for lidx, layer in enumerate(self.decoder.layers):
                    # emb_dim x ndim
                    alpha = np.squeeze(layer.mega_layer.move.alpha.cpu().numpy(), axis=-1)
                    ndim_mean = np.mean(alpha, axis=1)
                    edim_mean = np.mean(ndim_mean)
                    edim_std = np.std(ndim_mean)
                    edim_max, edim_min = np.max(ndim_mean), np.min(ndim_mean)

                    ndim_std = np.std(alpha, axis=1)
                    edim_std_mean = np.mean(ndim_std)
                    edim_std_std = np.std(ndim_std)
                    edim_std_max, edim_std_min = np.max(ndim_std), np.min(ndim_std)

                    logger.info("EMA Layer {}, mean of ndim--mean={:.5f},std={:.5f},max={:.5f},min={:.5f}, "
                                "std of ndim--mean={:.5f},std={:.5f},max={:.5f},min={:.5f}".format(lidx, edim_mean,
                                                                                                   edim_std, edim_max,
                                                                                                   edim_min, edim_std_mean,
                                                                                                   edim_std_std, edim_std_max,
                                                                                                   edim_std_min))
        if self.args.write_out_alpha and output:
            with torch.no_grad():
                for lidx, layer in enumerate(self.decoder.layers):
                    # emb_dim x ndim
                    alpha = np.squeeze(layer.mega_layer.move.alpha.cpu().numpy(), axis=-1)
                    np.savetxt(os.path.join(self.args.save_dir, "layer{}"), alpha, fmt='%.5f')


class MegaDecoderNoCrossAttn(FairseqIncrementalDecoder):
    """
    Mega decoder consisting of *args.decoder_layers* layers. Each layer is a :class:`MegaDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.embedding_dropout = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.chunk_size = args.decoder_chunk_size
        self.attention_activation = args.attention_activation_fn

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if embed_dim != input_embed_dim:
            self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False)
        else:
            self.project_in_dim = None

        norm_type = args.normalization_type
        normalize_embedding = getattr(args, 'normalize_embedding', False)
        normalize_before = getattr(args, 'normalize_before', False)
        assert not normalize_embedding or not normalize_before
        self.embed_norm = SequenceNorm(norm_type, embed_dim) if normalize_embedding else None

        self.layers = nn.ModuleList([])
        self.layers.extend([self.build_decoder_layer(args) for _ in range(args.decoder_layers)])
        self.num_layers = len(self.layers)

        final_norm_affine = True
        if hasattr(args, 'no_affine_final_norm'):
            final_norm_affine = not args.no_affine_final_norm
        self.final_norm = SequenceNorm(norm_type, embed_dim, affine=final_norm_affine) if normalize_before else None

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(self.embed_dim, len(dictionary), bias=False)
            nn.init.normal_(self.output_projection.weight, mean=0, std=self.embed_dim ** -0.5)

    def build_decoder_layer(self, args):
        return MegaDecoderLayer(args, no_cross_attention=True)

    def forward(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            incremental_state=incremental_state,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            incremental_state,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        bsz, seq_len = prev_output_tokens.size()
        if 0 < self.chunk_size < seq_len and seq_len % self.chunk_size != 0:
            num_paddings = math.ceil(seq_len / self.chunk_size) * self.chunk_size - seq_len
            prev_output_tokens = F.pad(prev_output_tokens, (0, num_paddings), value=self.padding_idx)
        else:
            num_paddings = 0

        # embed tokens
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if self.embed_norm is not None:
            x = self.embed_norm(x)

        x = self.embedding_dropout(x)

        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        if not decoder_padding_mask.any():
            decoder_padding_mask = None

        # account for padding while computing the representation
        if decoder_padding_mask is not None:
            # B x T
            inverse_mask = 1.0 - decoder_padding_mask.type_as(x)
            x = x * inverse_mask.unsqueeze(-1)
        else:
            inverse_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if x.size(0) > 1:
                attn_mask = self.buffered_future_mask(x)
            else:
                attn_mask = None

            x, layer_attn, _ = layer(x, incremental_state=incremental_state,
                                     attn_mask=attn_mask, decoder_padding_mask=decoder_padding_mask,
                                     need_attn=False)
            inner_states.append(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        if inverse_mask is not None:
            x = x * inverse_mask.transpose(0, 1).unsqueeze(-1)

        # remove padding tokens for chunk
        if num_paddings > 0:
            x = x[:seq_len]

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if 0 < self.chunk_size < dim:
            assert dim % self.chunk_size == 0
            dim = self.chunk_size
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            if self.attention_activation == 'softmax':
                self._future_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros(dim, dim)), 1)
            elif self.attention_activation in ['relu2', 'laplace']:
                self._future_mask = torch.tril(torch.ones(dim, dim), 0)
            else:
                raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("mega_lm", "mega_lm")
def base_lm_architecture(args):
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", 1664)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.decoder_hidden_dim)
    args.decoder_z_dim = getattr(args, 'decoder_z_dim', 128)
    args.decoder_n_dim = getattr(args, 'decoder_n_dim', 16)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_chunk_size = getattr(args, 'decoder_chunk_size', -1)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.hidden_dropout = getattr(args, "hidden_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.feature_dropout = getattr(args, 'feature_dropout', False)

    args.rel_pos_bias = getattr(args, 'rel_pos_bias', 'simple')
    args.normalization_type = getattr(args, 'normalization_type', 'layernorm')
    args.normalize_before = getattr(args, 'normalize_before', False)
    args.normalize_embedding = getattr(args, 'normalize_embedding', False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.no_affine_final_norm = getattr(args, 'no_affine_final_norm', False)

    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.adaptive_input_factor = getattr(args, 'adaptive_input_factor', 4)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', None)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)

    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.attention_activation_fn = getattr(args, 'attention_activation_fn', 'softmax')
    args.truncation_length = getattr(args, 'truncation_length', 2048)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)


@register_model_architecture('mega_lm', 'mega_lm_big')
def mega_lm_big(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 16)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_z_dim = getattr(args, 'decoder_z_dim', 256)
    base_lm_architecture(args)


@register_model_architecture('mega_lm', 'mega_lm_wiki103')
@register_model_architecture('mega_lm', 'mega_lm_adaptive_big')
def mega_lm_adaptive_big(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    args.adaptive_input = getattr(args, 'adaptive_input', True)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', True)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', '20000,60000')
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '20000,60000')
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0.2)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.hidden_dropout = getattr(args, 'hidden_dropout', 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', True)
    mega_lm_big(args)


@register_model_architecture('mega_lm', 'mega_lm_adaptive_base')
def mega_lm_adaptive_base(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 16)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", 1536)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1536)
    args.decoder_z_dim = getattr(args, 'decoder_z_dim', 128)

    args.dropout = getattr(args, 'dropout', 0.3)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.hidden_dropout = getattr(args, 'hidden_dropout', 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)

    args.adaptive_input = getattr(args, 'adaptive_input', True)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', True)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', '20000,60000')
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '20000,60000')
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0.2)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', True)
    base_lm_architecture(args)


@register_model_architecture('mega_lm', 'mega_lm_enwik8_base')
def mega_lm_adaptive_big_enwik8(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_z_dim = getattr(args, 'decoder_z_dim', 128)
    args.decoder_n_dim = getattr(args, 'decoder_n_dim', 16)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    args.hidden_dropout = getattr(args, 'hidden_dropout', 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    base_lm_architecture(args)


@register_model_architecture('mega_lm', 'mega_lm_enwik8_large')
def mega_lm_adaptive_large_enwik8(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_hidden_dim = getattr(args, "decoder_hidden_dim", 1536)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1536)
    args.decoder_z_dim = getattr(args, 'decoder_z_dim', 128)
    args.decoder_n_dim = getattr(args, 'decoder_n_dim', 16)
    args.dropout = getattr(args, 'dropout', 0.2)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.hidden_dropout = getattr(args, 'hidden_dropout', 0.1)
    base_lm_architecture(args)
