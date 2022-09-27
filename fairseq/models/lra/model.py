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
from fairseq.models.lra.transformer_lra_encoder import TransformerLRAEncoder
from fairseq.models.lra.luna_lra_encoder import LunaLRAEncoder
from fairseq.models.lra.lstm_lra_encoder import LSTMLRAEncoder
from fairseq.models.lra.flash_lra_encoder import FlashLRAEncoder
from fairseq.models.lra.mega_lra_encoder import MegaLRAEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model('lra')
class LRAModel(FairseqEncoderModel):
    """
    Class for training a transformer for LRA tasks.
    """
    def __init__(self, args, encoder, task):
        super().__init__(encoder)
        self.encoder = encoder
        self.args = args
        self.use_p = args.use_p
        self._max_positions = args.max_positions
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)

        self.classifier = nn.ModuleList([])
        if args.classifier_layers > 0:
            self.classifier.append(nn.Sequential(Linear(args.classifier_in_dim, args.classifier_out_dim), self.dropout_module))
            self.classifier.extend([
                nn.Sequential(Linear(args.classifier_out_dim, args.classifier_out_dim), self.dropout_module)
                for _ in range(args.classifier_layers - 1)
            ])
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
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float,
                            metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--act-dropout', type=float,
                            metavar='D', help='dropout probability after activation in FFN')
        parser.add_argument('--feature-dropout', action='store_true',
                            help='apply feature dropout')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--encoder-hidden-dim', type=int, metavar='N',
                            help='encoder hidden dimension for Mega')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--z-dim', type=int, metavar='N',
                            help='encoder z dimension for FLASH')
        parser.add_argument('--n-dim', type=int, metavar='N',
                            help='encoder n dimension for Mega')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        # Arguments related to input and output embeddings
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--share-encoder-input-output-embed',
                            action='store_true', help='share encoder input and output embeddings')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--no-token-positional-embeddings',
                            action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')

        parser.add_argument('--input-type', choices=['text', 'image'])
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--rel-pos-bias', choices=['simple', 'rotary'], default='simple')

        # Arguments related to sentence level prediction
        parser.add_argument('--sentence-class-num', type=int, metavar='N',
                            help='number of classes for sentence task')
        parser.add_argument('--sent-loss', action='store_true', help='if set, calculate sentence level predictions')

        # Arguments related to parameter initialization
        parser.add_argument('--apply-bert-init', action='store_true',
                            help='use custom param initialization for BERT')

        parser.add_argument('--use-p', default=False, action='store_true',
                            help='use p for prediction')

        # misc params
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--attention-activation-fn', choices=['softmax', 'relu2', 'laplace'],
                            help='activation function for attention mechanism')
        parser.add_argument('--classifier-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='Which activation function to use for classifier layer.')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')

        parser.add_argument('--layer-type', choices=['transformer', 'luna', 'lstm', 'flash', 'mega'])
        parser.add_argument('--norm-type', choices=['layernorm', 'scalenorm', 'rmsnorm', 'batchnorm', 'syncbatchnorm'])
        parser.add_argument('--normalize-embedding', action='store_true', help='normalize embedding for Mega.')
        parser.add_argument('--sen-rep-type', choices=['cls', 'mp'])

        parser.add_argument('--chunk-size', type=int, metavar='N',
                            help='chunk size of Mega.')
        parser.add_argument('--truncation-length', type=int, metavar='N',
                            help='truncation length of moving average layer.')
        parser.add_argument('--encoder-projection-length', type=int, metavar='N',
                            help='projected length of encoder as key')
        parser.add_argument('--encoder-projected-attention-heads', type=int, metavar='N',
                            help='num encoder projected attention heads')
        parser.add_argument('--decoder-projected-attention-heads', type=int, metavar='N',
                            help='num decoder projected attention heads')

    def forward(self, sample):
        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        sentence_rep = self.encoder(src_tokens, src_lengths)
        if not self.use_p:
            if self.layer_type in ['transformer', 'lstm', 'flash', 'mega']:
                sentence_rep = sentence_rep[1]
            elif self.layer_type == 'luna':
                sentence_rep = sentence_rep[1][0]
        else:
            sentence_rep = sentence_rep[1][1].mean(dim=0)
        if 'net_input1' in sample:
            src1_tokens = sample['net_input1']['src_tokens']
            src1_lengths = sample['net_input1']['src_lengths']
            sentence1_rep = self.encoder(src1_tokens, src1_lengths)
            if not self.use_p:
                if self.layer_type in ['transformer', 'lstm', 'flash', 'mega']:
                    sentence1_rep = sentence1_rep[1]
                elif self.layer_type == 'luna':
                    sentence1_rep = sentence1_rep[1][0]
            else:
                sentence1_rep = sentence1_rep[1][1].mean(dim=0)
            concat_rep = []
            concat_rep.append(sentence1_rep)
            concat_rep.append(sentence_rep)
            sentence_rep = torch.cat(concat_rep, dim=-1)
        for layer in self.classifier:
            sentence_rep = self.classifier_activation(layer(sentence_rep))
        if self.sentence_projection_layer:
            sentence_logits = self.sentence_projection_layer(sentence_rep)
        return {
            'encoder_out': sentence_logits
        }

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
        if not hasattr(args, 'decoder_embed_dim'):
            args.decoder_embed_dim = args.encoder_embed_dim
        encoder = LRAEncoder(args, task)
        return cls(args, encoder, task)


class LRAEncoder(FairseqEncoder):
    """LRA encoder."""

    def __init__(self, args, task):
        if args.input_type == 'text':
            dictionary = task.dictionary
            vocab_size = len(dictionary)
            padding_idx = dictionary.pad_index
            offset_positions_by_padding = True
            embedding_type = 'sparse'
        else:
            assert args.sen_rep_type == 'mp' or args.layer_type == 'lstm'
            dictionary = None
            vocab_size = None
            padding_idx = None
            offset_positions_by_padding = False
            embedding_type = 'linear'
        super().__init__(dictionary)
        self.args = args
        if args.layer_type == 'transformer':
            self.encoder = TransformerLRAEncoder(
                tie_layer_weights=getattr(args, 'tie_layer_weights', False),
                padding_idx=padding_idx,
                vocab_size=vocab_size,
                num_encoder_layers=args.encoder_layers,
                embedding_type=embedding_type,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                max_seq_len=args.max_positions,
                use_position_embeddings=True,
                offset_positions_by_padding=offset_positions_by_padding,
                encoder_normalize_before=getattr(args, "encoder_normalize_before", False),
                apply_bert_init=getattr(args, "apply_bert_init", False),
                activation_fn=args.activation_fn,
                learned_pos_embedding=args.encoder_learned_pos,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls')
            )
        elif args.layer_type == 'lstm':
            self.encoder = LSTMLRAEncoder(
                padding_idx=padding_idx,
                vocab_size=vocab_size,
                num_layers=args.encoder_layers,
                bidirectional=True,
                embedding_type=embedding_type,
                embedding_dim=args.encoder_embed_dim,
                hidden_dim=args.encoder_ffn_embed_dim,
                input_dropout=args.dropout,
                output_dropout=args.act_dropout,
                max_seq_len=args.max_positions,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls')
            )
        elif args.layer_type == 'flash':
            self.encoder = FlashLRAEncoder(
                padding_idx=padding_idx,
                vocab_size=vocab_size,
                num_encoder_layers=args.encoder_layers,
                embedding_type=embedding_type,
                embedding_dim=args.encoder_embed_dim,
                hidden_dim=args.encoder_hidden_dim,
                z_dim=args.z_dim,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                hidden_dropout=args.act_dropout,
                norm_type=args.norm_type,
                max_seq_len=args.max_positions,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls')
            )
        elif args.layer_type == 'mega':
            self.encoder = MegaLRAEncoder(
                padding_idx=padding_idx,
                vocab_size=vocab_size,
                num_encoder_layers=args.encoder_layers,
                embedding_type=embedding_type,
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
                normalize_embedding=args.normalize_embedding,
                feature_dropout=args.feature_dropout,
                chunk_size=getattr(args, 'chunk_size', -1),
                truncation=getattr(args, 'truncation_length', None),
                rel_pos_bias=args.rel_pos_bias,
                max_seq_len=args.max_positions,
                sen_rep_type=getattr(args, 'sen_rep_type', 'mp')
            )
        else:
            self.encoder = LunaLRAEncoder(
                tie_layer_weights=getattr(args, 'tie_layer_weights', False),
                projection_length=args.encoder_projection_length,
                padding_idx=padding_idx,
                vocab_size=vocab_size,
                num_encoder_layers=args.encoder_layers,
                embedding_type=embedding_type,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                num_projected_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                max_seq_len=args.max_positions,
                use_position_embeddings=True,
                offset_positions_by_padding=offset_positions_by_padding,
                layernorm_embedding=getattr(args, "encoder_normalize_before", False),
                normalize_before=False,
                apply_bert_init=getattr(args, "apply_bert_init", False),
                tie_kv=getattr(args, 'tie_kv', False),
                activation_fn=args.activation_fn,
                learned_pos_embedding=args.encoder_learned_pos,
                embed_scale=None,
                sen_rep_type=getattr(args, 'sen_rep_type', 'cls')
            )

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        return self.encoder(src_tokens, src_lengths, last_state_only=True)


@register_model_architecture('lra', 'lra')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.act_dropout = getattr(args, 'act_dropout', 0.0)
    args.feature_dropout = getattr(args, 'feature_dropout', False)

    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 2048)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', args.encoder_hidden_dim)
    args.z_dim = getattr(args, 'z_dim', 128)
    args.n_dim = getattr(args, 'n_dim', 2)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 2048)

    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.sent_loss = getattr(args, 'sent_loss', True)

    args.apply_bert_init = getattr(args, 'apply_bert_init', True)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.attention_activation_fn = getattr(args, 'attention_activation_fn', 'relu2')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_activation_fn = getattr(args, 'classifier_activation_fn', 'gelu')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.normalize_embedding = getattr(args, 'normalize_embedding', False)
    args.layer_type = getattr(args, 'layer_type', 'transformer')
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.classifier_in_dim = getattr(args, "classifier_in_dim", args.encoder_ffn_embed_dim * 2 if args.layer_type == 'lstm' else args.encoder_embed_dim)


@register_model_architecture('lra', 'transformer_lra_listop')
def transformer_lra_listop(args):
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 2002)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.tie_layer_weights = getattr(args, 'tie_layer_weights', True)
    base_architecture(args)


@register_model_architecture('lra', 'luna_lra_listop')
def luna_lra_listop(args):
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 2002)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.tie_layer_weights = getattr(args, 'tie_layer_weights', True)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    base_architecture(args)


@register_model_architecture('lra', 'mega_lra_listop')
def mega_lra_listop(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'mega')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 160)
    args.z_dim = getattr(args, 'z_dim', 64)
    args.n_dim = getattr(args, 'n_dim', 16)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 80)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 160)
    args.chunk_size = getattr(args, 'chunk_size', -1)
    args.truncation_length = getattr(args, 'truncation_length', 1024)
    args.max_positions = getattr(args, 'max_positions', 2002)
    args.norm_type = getattr(args, 'norm_type', 'scalenorm')
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)


@register_model_architecture('lra', 'transformer_lra_imdb')
def transformer_lra_imdb_architecture(args):
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 1024)
    base_architecture(args)


@register_model_architecture('lra', 'luna_lra_imdb')
def luna_lra_imdb_architecture(args):
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_imdb_architecture(args)


@register_model_architecture('lra', 'flash_lra_imdb')
def flash_lra_imdb(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'flash')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.z_dim = getattr(args, 'z_dim', 64)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.norm_type = getattr(args, 'norm_type', 'scalenorm')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.max_positions = getattr(args, 'max_positions', 4002)
    base_architecture(args)


@register_model_architecture('lra', 'mega_lra_imdb')
def mega_lra_imdb(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'mega')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.z_dim = getattr(args, 'z_dim', 64)
    args.n_dim = getattr(args, 'n_dim', 16)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.chunk_size = getattr(args, 'chunk_size', -1)
    args.truncation_length = getattr(args, 'truncation_length', 1024)
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.norm_type = getattr(args, 'norm_type', 'scalenorm')
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)


@register_model_architecture('lra', 'transformer_lra_aan')
def transformer_lra_aan_architecture(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 512)
    args.classifier_in_dim = getattr(args, 'classifier_in_dim', args.encoder_embed_dim * 2)
    base_architecture(args)


@register_model_architecture('lra', 'luna_lra_aan')
def luna_lra_aan_architecture(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_aan_architecture(args)


@register_model_architecture('lra', 'mega_lra_aan')
def mega_lra_aan(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'mega')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.z_dim = getattr(args, 'z_dim', 64)
    args.n_dim = getattr(args, 'n_dim', 16)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.classifier_in_dim = getattr(args, 'classifier_in_dim', args.encoder_embed_dim * 2)
    args.chunk_size = getattr(args, 'chunk_size', -1)
    args.truncation_length = getattr(args, 'truncation_length', 1024)
    args.max_positions = getattr(args, 'max_positions', 4002)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)


@register_model_architecture('lra', 'transformer_lra_cifar10')
def transformer_lra_cifar10(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 128)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    base_architecture(args)


@register_model_architecture('lra', 'luna_lra_cifar10')
def luna_lra_cifar10(args):
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_cifar10(args)


@register_model_architecture('lra', 'flash_lra_cifar10')
def flash_lra_cifar10(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'flash')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 512)
    args.z_dim = getattr(args, 'z_dim', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 192)
    args.norm_type = getattr(args, 'norm_type', 'batchnorm')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 512)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.max_positions = getattr(args, 'max_positions', 1024)
    base_architecture(args)


@register_model_architecture('lra', 'mega_lra_cifar10')
def mega_lra_cifar10(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'mega')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 320)
    args.z_dim = getattr(args, 'z_dim', 96)
    args.n_dim = getattr(args, 'n_dim', 16)
    args.encoder_layers = getattr(args, 'encoder_layers', 8)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 160)
    args.norm_type = getattr(args, 'norm_type', 'batchnorm')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 320)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.chunk_size = getattr(args, 'chunk_size', 1024)
    args.truncation_length = getattr(args, 'truncation_length', 1024)
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)


@register_model_architecture('lra', 'transformer_lra_pf32')
def transformer_lra_pf32(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.max_positions = getattr(args, 'max_positions', 1026)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)


@register_model_architecture('lra', 'luna_lra_pf32')
def luna_lra_pf32(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_pf32(args)


@register_model_architecture('lra', 'flash_lra_pf32')
def flash_lra_pf32(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'flash')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 384)
    args.z_dim = getattr(args, 'z_dim', 64)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.norm_type = getattr(args, 'norm_type', 'batchnorm')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 384)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.max_positions = getattr(args, 'max_positions', 1024)
    base_architecture(args)


@register_model_architecture('lra', 'mega_lra_pf32')
def mega_lra_pf32(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'mega')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.z_dim = getattr(args, 'z_dim', 64)
    args.n_dim = getattr(args, 'n_dim', 16)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.norm_type = getattr(args, 'norm_type', 'batchnorm')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 256)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.chunk_size = getattr(args, 'chunk_size', 1024)
    args.truncation_length = getattr(args, 'truncation_length', 1024)
    args.max_positions = getattr(args, 'max_positions', 1024)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)


@register_model_architecture('lra', 'luna_lra_pf128')
def luna_lra_pf128(args):
    args.max_positions = getattr(args, 'max_positions', 128 * 128 + 2)
    args.layer_type = getattr(args, 'layer_type', 'luna')
    transformer_lra_pf32(args)


@register_model_architecture('lra', 'mega_lra_pf128')
def mega_lra_pf128(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    args.layer_type = getattr(args, 'layer_type', 'mega')
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 128)
    args.z_dim = getattr(args, 'z_dim', 32)
    args.n_dim = getattr(args, 'n_dim', 16)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.activation_fn = getattr(args, 'activation_fn', 'silu')
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.norm_type = getattr(args, 'norm_type', 'batchnorm')
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 128)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 2)
    args.chunk_size = getattr(args, 'chunk_size', 128 * 128)
    args.truncation_length = getattr(args, 'truncation_length', 4096)
    args.max_positions = getattr(args, 'max_positions', 128 * 128)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')
    base_architecture(args)
