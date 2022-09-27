"""
Luna Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    LunaSentenceEncoder
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .hub_interface import LunaHubInterface

logger = logging.getLogger(__name__)


@register_model('luna_bert')
class LunaBertModel(FairseqEncoderModel):

    @classmethod
    def hub_models(cls):
        return None

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        # args for Luna
        parser.add_argument('--projection-length', type=int, help='Luna projection length')
        parser.add_argument('--fix-projection-length', action='store_true',
                            help='fix projection length for all input sequences')
        parser.add_argument('--untie-luna-kv', action='store_true',
                            help='Untie key and value parameters in Luna attention')
        # args for FFN
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        # args for dropout
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        parser.add_argument('--untie-weights-luna', action='store_true',
                            help='Untie weights between embeddings and classifiers in Luna')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        encoder = LunaBertEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(self, src_tokens, features_only=False, return_packed_features=False, return_all_hiddens=False,
                classification_head_name=None, **kwargs):
        if classification_head_name is not None:
            features_only = True

        x, px, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if return_packed_features:
            extra['packed_features'] = px

        if classification_head_name is not None:
            px_padding_mask = extra['padding_masks'][1]
            x = self.classification_heads[classification_head_name](x, px, px_padding_mask=px_padding_mask)
            return x, extra
        else:
            return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        if name == 'pooling_classification_head':
            self.classification_heads[name] = LunaPoolingClassificationHead(
                self.args.encoder_embed_dim,
                inner_dim or self.args.encoder_embed_dim,
                num_classes,
                self.args.pooler_activation_fn,
                self.args.pooler_dropout,
                self.args.quant_noise_pq,
                self.args.quant_noise_pq_block_size,
            )
        elif name == 'attention_classification_head':
            self.classification_heads[name] = LunaAttentionClassificationHead(
                self.args.encoder_embed_dim,
                inner_dim or self.args.encoder_embed_dim,
                num_classes,
                self.args.encoder_attention_heads,
                self.args.pooler_activation_fn,
                self.args.pooler_dropout,
                self.args.quant_noise_pq,
                self.args.quant_noise_pq_block_size,
            )
        else:
            self.classification_heads[name] = LunaCLSClassificationHead(
                self.args.encoder_embed_dim,
                inner_dim or self.args.encoder_embed_dim,
                num_classes,
                self.args.pooler_activation_fn,
                self.args.pooler_dropout,
                self.args.quant_noise_pq,
                self.args.quant_noise_pq_block_size,
            )

    @property
    def supported_targets(self):
        return {'self'}

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return LunaHubInterface(x['args'], x['task'], x['models'][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + 'decoder'):
                new_k = prefix + 'encoder' + k[len(prefix + 'decoder'):]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            [] if not hasattr(self, 'classification_heads')
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v


class LunaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class LunaCLSClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout, q_noise=0, qn_block_size=8):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )

    def forward(self, x, px, **kwargs):
        x = x[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LunaPoolingClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout, q_noise=0, qn_block_size=8):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, input_dim)
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )

    def forward(self, x, px, px_padding_mask=None, **kwargs):
        # B x L x C
        px = self.in_proj(px)
        # B x C
        if px_padding_mask is not None:
            px = px.masked_fill(px_padding_mask.unsqueeze(2), float('-inf'))
        x, _ = px.max(dim=1)

        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LunaAttentionClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, num_heads, activation_fn, pooler_dropout, q_noise=0, qn_block_size=8):
        super().__init__()
        self.attn = MultiheadAttention(
            input_dim,
            num_heads,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.layernorm = LayerNorm(input_dim)
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )

    def forward(self, x, px, px_padding_mask=None, **kwargs):
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # 1 x B x C
        x = x[:1] # take <s> token (equiv. to [CLS])
        # B x L x C -> L x B x C
        px = px.transpose(0, 1)
        # 1 x B x C
        x, _ = self.attn(query=x, key=px, value=px, key_padding_mask=px_padding_mask)
        x = self.layernorm(x)
        # B x C
        x = x.squeeze(0)

        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LunaBertEncoder(FairseqEncoder):
    """Luna encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        self.sentence_encoder = LunaSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            projection_length=args.projection_length,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            num_projected_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            normalize_before=False,
            layernorm_embedding=True,
            dynamic_projection=not args.fix_projection_length,
            tie_kv=not args.untie_luna_kv,
            max_seq_len=args.max_positions,
            num_segments=0,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
        )

        args.untie_weights_luna = getattr(args, 'untie_weights_luna', False)

        self.lm_head = LunaLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_tokens.weight if not args.untie_weights_luna else None,
        )

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - the projected output of shape `(batch, plen, embed_dim)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, px, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, px, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _, padding_masks = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
        )

        features = inner_states[-1]
        x = features[0].transpose(0, 1)  # T x B x C -> B x T x C
        px = features[1].transpose(0, 1) # L x B x C -> B x L x C
        return x, px, {'inner_states': inner_states if return_all_hiddens else None,
                       'padding_masks': padding_masks}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


@register_model_architecture('luna_bert', 'luna_bert')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.max_positions = getattr(args, 'max_positions', 512)
    args.projection_length = getattr(args, 'projection_length', 128)
    args.fix_projection_length = getattr(args, "fix_projection_length", False)
    args.untie_luna_kv = getattr(args, "untie_luna_kv", False)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    args.encoder_layers_to_keep = getattr(args, 'encoder_layers_to_keep', None)
    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0.0)


@register_model_architecture('luna_bert', 'luna_base_512')
def luna_base_architecture_512(args):
    base_architecture(args)


@register_model_architecture('luna_bert', 'luna_base_tied_512')
def luna_base_architecture_512(args):
    args.untie_luna_kv = getattr(args, "untie_luna_kv", False)
    base_architecture(args)


@register_model_architecture('luna_bert', 'luna_base_untied_512')
def luna_base_architecture_512(args):
    args.untie_luna_kv = getattr(args, "untie_luna_kv", True)
    base_architecture(args)


@register_model_architecture('luna_bert', 'luna_base_2048')
def luna_base_architecture_2048(args):
    args.max_positions = getattr(args, 'max_positions', 2048)
    args.projection_length = getattr(args, 'projection_length', 256)
    base_architecture(args)


@register_model_architecture('luna_bert', 'luna_large_512')
def luna_large_architecture_512(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    base_architecture(args)


@register_model_architecture('luna_bert', 'luna_large_2048')
def luna_large_architecture_2048(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.projection_length = getattr(args, 'projection_length', 256)
    base_architecture(args)
