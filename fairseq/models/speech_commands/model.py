# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoderModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    FairseqDropout,
)
from fairseq.models.speech_commands.mega_scraw_encoder import MegaSCRawEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model('sc_raw')
class SCRawModel(FairseqEncoderModel):
    """
    Class for training a transformer for LRA tasks.
    """
    def __init__(self, args, encoder, task):
        super().__init__(encoder)
        self.encoder = encoder
        self.args = args
        self._max_positions = args.max_positions
        self.sentence_out_dim = args.sentence_class_num
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.classifier = nn.ModuleList([])
        self.classifier.append(nn.Sequential(Linear(args.classifier_in_dim, args.classifier_out_dim),
                                             self.dropout_module))
        self.classifier.extend([
            nn.Sequential(Linear(args.classifier_out_dim, args.classifier_out_dim), self.dropout_module)
            for _ in range(args.classifier_layers - 1)
        ])
        # self.classifier = nn.Linear(args.classifier_in_dim, args.classifier_out_dim)
        self.classifier_activation = utils.get_activation_fn(args.classifier_activation_fn)
        self.sentence_projection_layer = Linear(
            args.classifier_out_dim,
            self.sentence_out_dim,
            bias=False
        )
        self.sen_rep_type = getattr(args, "sen_rep_type", "cls")
        self.layer_type = args.layer_type

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, 'apply_bert_init', False):
            self.apply(init_bert_params)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--act-dropout', type=float, metavar='D', help='dropout probability after activation in FFN')
        parser.add_argument('--feature-dropout', action='store_true', help='apply feature dropout')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--encoder-hidden-dim', type=int, metavar='N', help='encoder hidden dimension for Mega')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--z-dim', type=int, metavar='N', help='encoder z dimension for FLASH')
        parser.add_argument('--n-dim', type=int, metavar='N', help='encoder n dimension for Mega')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')

        # Arguments related to input and output embeddings
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--max-positions', type=int, help='number of positional embeddings to learn')
        parser.add_argument('--rel-pos-bias', choices=['simple', 'rotary'], default='simple')

        # Arguments related to sentence level prediction
        parser.add_argument('--sentence-class-num', type=int, metavar='N', help='number of classes for sentence task')
        parser.add_argument('--sent-loss', action='store_true', help='if set, calculate sentence level predictions')

        # Arguments related to parameter initialization
        parser.add_argument('--apply-bert-init', action='store_true', help='use custom param initialization for BERT')

        # misc params
        parser.add_argument('--activation-fn', choices=utils.get_available_activation_fns(), help='activation function to use')
        parser.add_argument('--attention-activation-fn', choices=['softmax', 'relu2', 'laplace'], help='activation function for attention mechanism')
        parser.add_argument('--classifier-activation-fn', choices=utils.get_available_activation_fns(),
                            help='Which activation function to use for classifier layer.')
        parser.add_argument('--encoder-normalize-before', action='store_true', help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0, help='LayerDrop probability for encoder')

        parser.add_argument('--layer-type', choices=['mega'])
        parser.add_argument('--norm-type', choices=['layernorm', 'scalenorm', 'rmsnorm', 'batchnorm', 'syncbatchnorm'])
        parser.add_argument('--normalize-embedding', action='store_true', help='normalize embedding for Mega.')
        parser.add_argument('--sen-rep-type', choices=['cls', 'mp'])

        parser.add_argument('--chunk-size', type=int, metavar='N',help='chunk size of Mega.')
        parser.add_argument('--truncation-length', type=int, metavar='N', help='truncation length of moving average layer.')

    def forward(self, sample):
        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        sentence_rep = self.encoder(src_tokens, src_lengths)
        sentence_rep = sentence_rep[1]

        for layer in self.classifier:
            sentence_rep = self.classifier_activation(layer(sentence_rep))

        sentence_logits = self.sentence_projection_layer(sentence_rep)
        return {'encoder_out': sentence_logits}

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self._max_positions

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.max_positions

        encoder = SCRawEncoder(args, task)
        return cls(args, encoder, task)


class SCRawEncoder(FairseqEncoder):
    """LRA encoder."""

    def __init__(self, args, task):
        assert args.sen_rep_type == 'mp' or args.layer_type == 'lstm'
        super().__init__(None)
        self.args = args
        if args.layer_type == 'mega':
            self.encoder = MegaSCRawEncoder(
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                hidden_dim=args.encoder_hidden_dim,
                ffn_hidden_dim=args.encoder_ffn_embed_dim,
                z_dim=args.z_dim,
                n_dim=args.n_dim,
                activation=args.activation_fn,
                attention_activation=args.attention_activation_fn,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                hidden_dropout=args.act_dropout,
                norm_type=args.norm_type,
                normalize_before=args.encoder_normalize_before,
                feature_dropout=args.feature_dropout,
                chunk_size=getattr(args, 'chunk_size', -1),
                truncation=getattr(args, 'truncation_length', None),
                rel_pos_bias=args.rel_pos_bias,
                max_seq_len=args.max_positions,
                sen_rep_type=getattr(args, 'sen_rep_type', 'mp')
            )

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        return self.encoder(src_tokens, src_lengths, last_state_only=True)


@register_model_architecture('sc_raw', 'sc_raw')
def base_architecture(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'mega')

    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 60)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 120)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', args.encoder_hidden_dim)
    args.z_dim = getattr(args, 'z_dim', 30)
    args.n_dim = getattr(args, 'n_dim', 16)

    args.dropout = getattr(args, 'dropout', 0.0)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    args.act_dropout = getattr(args, 'act_dropout', 0.0)
    args.feature_dropout = getattr(args, 'feature_dropout', False)

    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.attention_activation_fn = getattr(args, 'attention_activation_fn', 'laplace')
    args.norm_type = getattr(args, 'norm_type', 'batchnorm')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)


    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 2 * args.encoder_embed_dim)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.classifier_activation_fn = getattr(args, 'classifier_activation_fn', 'gelu')
    args.classifier_in_dim = getattr(args, "classifier_in_dim", args.encoder_embed_dim)
    args.sent_loss = getattr(args, 'sent_loss', True)

    args.chunk_size = getattr(args, 'chunk_size', 1000)
    args.truncation_length = getattr(args, 'truncation_length', 1000)
    args.max_positions = getattr(args, 'max_positions', 16000)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')


@register_model_architecture('sc_raw', 'mega_sc_raw')
def mega_lra_sc(args):
    base_architecture(args)


@register_model_architecture('sc_raw', 'mega_sc_raw_base')
def mega_lra_sc_base(args):
    mega_lra_sc(args)


@register_model_architecture('sc_raw', 'mega_sc_raw_big')
def mega_lra_sc_big(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 72)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 144)
    args.z_dim = getattr(args, 'z_dim', 36)
    base_architecture(args)
