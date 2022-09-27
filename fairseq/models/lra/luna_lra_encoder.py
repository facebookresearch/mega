from typing import Optional, Tuple, List, Union
import math

import torch
import torch.nn as nn
from torch.nn import Parameter

from fairseq.modules import (
    LayerNorm,
    LayerDropModuleList,
    PositionalEmbedding,
    RealNumberEmbedding,
)
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.luna_sentence_encoder import LunaSentenceEncoderLayer, init_bert_params, get_sinusoidal_positional_embedding


class LunaLRAEncoder(nn.Module):
    """
    Implementation for a Bi-directional Luna based Sentence Encoder used
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
        projection_length: int,
        num_encoder_layers: int = 6,
        embedding_type: str = "sparse",
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        num_projected_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        layernorm_embedding: bool = False,
        normalize_before: bool = False,
        dynamic_projection: bool = True,
        tie_kv=True,
        tie_layer_weights: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        sen_rep_type: str = 'cls',
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.proj_len = projection_length
        self.dynamic_projection = dynamic_projection
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tie_layer_weights = tie_layer_weights
        self.tpu = False  # whether we're on TPU
        self.sen_rep_type = sen_rep_type

        assert embedding_type in ['sparse', 'linear']
        self.embed_tokens = self.build_embedding(self.embedding_type, self.vocab_size, self.embedding_dim, self.padding_idx)
        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), q_noise, qn_block_size)
        else:
            self.quant_noise = None

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        self.projected_embeddings = Parameter(torch.Tensor(self.proj_len, self.embedding_dim))
        nn.init.normal_(self.projected_embeddings, mean=0.0, std=self.embedding_dim ** -0.5)
        if self.use_position_embeddings and not self.learned_pos_embedding:
            projected_positions = get_sinusoidal_positional_embedding(self.proj_len, self.embedding_dim)
            if self.embed_scale is None:
                self.embed_scale = math.sqrt(self.embedding_dim)
        else:
            projected_positions = None
        self.register_buffer('projected_positions', projected_positions)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        self.num_layers = num_encoder_layers
        real_num_layes = 1 if self.tie_layer_weights else num_encoder_layers
        self.layers.extend([
            self.build_luna_sentence_encoder_layer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                num_projected_attention_heads=num_projected_attention_heads,
                dropout=self.dropout_module.p,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                normalize_before=normalize_before,
                tie_kv=tie_kv,
                export=export,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )
            for _ in range(real_num_layes)
        ])

        assert not layernorm_embedding or not normalize_before

        if layernorm_embedding:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
            self.proj_emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None
            self.proj_emb_layer_norm = None

        if normalize_before:
            self.layer_norm = LayerNorm(self.embedding_dim, export=export)
            self.proj_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.layer_norm = None
            self.proj_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

    def build_embedding(self, embedding_type, vocab_size, embedding_dim, padding_idx):
        if embedding_type == 'sparse':
            embed_tokens = nn.Embedding(vocab_size, embedding_dim, padding_idx)
            nn.init.normal_(embed_tokens.weight, mean=0, std=embedding_dim ** -0.5)
            return embed_tokens
        else:
            embed_tokens = RealNumberEmbedding(embedding_dim)
            return embed_tokens

    def build_luna_sentence_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        num_projected_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        normalize_before,
        tie_kv,
        export,
        q_noise,
        qn_block_size,
    ):
        return LunaSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            num_projected_attention_heads=num_projected_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            normalize_before=normalize_before,
            tie_kv=tie_kv,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(
        self,
        tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]],
               Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor]]:

        # compute padding mask. This is needed for multi-head attention
        # B x T
        if self.embedding_type == 'sparse':
            x_padding_mask = tokens.eq(self.padding_idx)
            if not self.traceable and not self.tpu:
                if not x_padding_mask.any():
                    x_padding_mask = None
            # B x T -> B x T x D
            x = self.embed_tokens(tokens)
        else:
            x_padding_mask = None
            # B x T -> B x T x 1 -> B x T x D
            x = self.embed_tokens(tokens)

        lengths = tokens.size(1)
        if x_padding_mask is not None:
            lengths = lengths - x_padding_mask.sum(1)
        max_len = lengths.max() if self.dynamic_projection else self.proj_len

        px = self.projected_embeddings[:max_len]

        if self.embed_scale is not None:
            x *= self.embed_scale
            px *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)
        if self.projected_positions is not None:
            px += self.projected_positions[:max_len]

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
            px = self.proj_emb_layer_norm(px)

        bsz = x.size(0)
        len, dim = px.size()
        # L x C -> B x L x C
        px = px.unsqueeze(0).expand(bsz, len, dim)

        if self.dynamic_projection and self.embedding_type == 'sparse':
            pidx = torch.arange(len).unsqueeze(0).to(x.device)
            # B x L
            px_padding_mask = pidx.ge(lengths.unsqueeze(1))
            if not self.traceable and not self.tpu and not px_padding_mask.any():
                px_padding_mask = None
        else:
            px_padding_mask = None

        x = self.dropout_module(x)
        px = self.dropout_module(px)

        # account for padding while computing the representation
        if x_padding_mask is not None:
            x = x * (1 - x_padding_mask.unsqueeze(-1).type_as(x))
        if px_padding_mask is not None:
            px = px * (1 - px_padding_mask.unsqueeze(-1).type_as(px))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # B x L x C -> L x B x C
        px = px.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append((x, px))

        for i in range(self.num_layers):
            if self.tie_layer_weights:
                x, px, _ = self.layers[0](x, px,
                             x_padding_mask=x_padding_mask,
                             px_padding_mask=px_padding_mask)
            else:
                x, px, _ = self.layers[i](x, px,
                             x_padding_mask=x_padding_mask,
                             px_padding_mask=px_padding_mask)
            if not last_state_only:
                inner_states.append((x, px))

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            px = self.proj_layer_norm(px)

        if x_padding_mask is not None:
            x = x * (1 - x_padding_mask.transpose(0, 1).unsqueeze(-1).type_as(x))

        if self.sen_rep_type == 'cls':
            sentence_rep = x[0, :, :]
        elif self.sen_rep_type == 'mp':
            sentence_rep = x.sum(dim=0) / src_lengths.unsqueeze(1)
        sentence_proj_rep = px

        if last_state_only:
            inner_states = [(x, px)]

        return inner_states, (sentence_rep, sentence_proj_rep), (x_padding_mask, px_padding_mask)