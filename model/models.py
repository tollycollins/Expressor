"""
Expressor Transformer model

References:
    https://github.com/YatingMusic/compound-word-transformer/blob/main/workspace/uncond/cp-linear/main-cp.py
    https://fast-transformers.github.io/
    https://fast-transformers.github.io/api_docs/fast_transformers/transformers.html
    https://github.com/YatingMusic/MuseMorphose/blob/main/model/musemorphose.py
"""
import numpy as np

import torch
import torch.nn as nn

from fast_transformers.transformers import TransformerEncoderLayer, TransformerDecoderLayer
from fast_transformers.recurrent.transformers import RecurrentTransformerDecoderLayer
from fast_transformers.attention import AttentionLayer, LinearAttention

from modules import (
    Embedding, PositionalEncoding, EncoderBlock, DecoderBlock, RecurrentDecoderBlock
)
from helpers import weights_init


class Expressor(nn.Module):
    """
    Parallel encoder-decoder blocks with optional skip connections.  
    Generation-guiding tokens can be added to cross-attention latent space.  
    """
    def __init__(self, enc_t_types, out_t_types,
                 enc_vocab_sizes, enc_emb_dims,
                 enc_dim, enc_n_layers, enc_heads, enc_ff_dim,
                 dec_vocab_sizes, dec_emb_dims,
                 dec_dim, dec_n_layers, dec_heads, dec_ff_dim,
                 attr_types=[], attr_vocab_sizes=[], attr_emb_dims=[],
                 attr_pos_emb=False,
                 enc_norm_layer=True,
                 enc_dropout=0.1, enc_act='relu',
                 dec_dropout=0.1, dec_act='relu',
                 skips=True,
                 hidden=True,
                 is_training=True):
        super().__init__()

        # record hyperparameters
        self.enc_t_types = enc_t_types
        self.attr_types = attr_types
        self.out_t_types = out_t_types

        self.enc_vocab_sizes = enc_vocab_sizes
        self.enc_emb_dims = enc_emb_dims
        self.enc_dim = enc_dim
        self.enc_n_layers = enc_n_layers
        self.enc_heads = enc_heads
        self.enc_ff_dim = enc_ff_dim
        self.enc_norm_layer = enc_norm_layer
        self.enc_dropout = enc_dropout
        self.enc_act = enc_act

        self.attr_vocab_sizes = attr_vocab_sizes
        self.attr_emb_dims = attr_emb_dims
        self.attr_pos_emb = attr_pos_emb

        self.dec_vocab_sizes = dec_vocab_sizes
        self.dec_emb_dims = dec_emb_dims
        self.dec_dim = dec_dim
        self.dec_n_layers = dec_n_layers
        self.dec_heads = dec_heads
        self.dec_ff_dim = dec_ff_dim
        self.dec_dropout = dec_dropout
        self.dec_act = dec_act

        self.skips = skips
        self.hidden = hidden

        self.is_training = is_training

        # embeddings
        self.enc_embeddings = nn.ModuleList([
            Embedding(v, e) for v, e in zip(enc_vocab_sizes, enc_emb_dims)
        ])
        if len(attr_types):
            self.attr_embeddings = nn.ModuleList([
                Embedding(v, e) for v, e in zip(attr_vocab_sizes, attr_emb_dims)
            ])
        self.dec_embeddings = nn.ModuleList([
            Embedding(v, e) for v, e in zip(dec_vocab_sizes, dec_emb_dims)
        ])
        
        # linear for embeddings
        self.enc_emb_lin = nn.Linear(np.sum(enc_emb_dims), enc_dim)
        self.dec_emb_lin = nn.Linear(np.sum(dec_emb_dims), dec_dim)

        # latent dimensions
        attr_emb_dim = 0 if attr_emb_dims == [] else np.sum(attr_emb_dims)
        # self.attr_emb_lin = nn.Linear(attr_emb_dim + enc_dim, dec_dim)
        
        # positional encoding
        self.enc_pos = PositionalEncoding(enc_dim, dropout=enc_dropout)
        self.dec_pos = PositionalEncoding(dec_dim, dropout=dec_dropout)
        if attr_pos_emb:
            self.attr_pos = PositionalEncoding(attr_emb_dim, dropout=0)
        
        # encoder
        self.enc_block = EncoderBlock([
            TransformerEncoderLayer(
                AttentionLayer(LinearAttention(enc_dim), enc_dim, enc_heads), 
                enc_dim, 
                d_ff=enc_ff_dim,
                dropout=enc_dropout,
                activation=enc_act
            ) for l in range(enc_n_layers)
        ], nn.LayerNorm(enc_dim) if enc_norm_layer else None)

        # decoder
        if is_training:
            # parallel version with teacher forcing for training
            self.dec_block = DecoderBlock([
                TransformerDecoderLayer(
                    AttentionLayer(LinearAttention(dec_dim), dec_dim, dec_heads), 
                    AttentionLayer(LinearAttention(dec_dim), dec_dim, dec_heads, 
                                   d_keys=attr_emb_dim // dec_heads, 
                                   d_values=attr_emb_dim // dec_heads),
                    dec_dim,
                    d_ff=dec_ff_dim,
                    dropout=dec_dropout,
                    activation=dec_act
                ) for l in range(dec_n_layers)
            ])
        else:
            # autoregressive version for inference
            self.dec_block = RecurrentDecoderBlock([
                RecurrentTransformerDecoderLayer(
                    AttentionLayer(LinearAttention(dec_dim), dec_dim, dec_heads), 
                    AttentionLayer(LinearAttention(dec_dim), dec_dim, dec_heads, 
                                   d_keys=attr_emb_dim // dec_heads, 
                                   d_values=attr_emb_dim // dec_heads),
                    dec_dim,
                    d_ff=dec_ff_dim,
                    dropout=dec_dropout,
                    activation=dec_act
                ) for l in range(dec_n_layers)
            ])
        
        # type residual connection
        if out_t_types[0] == 'type':
            self.res_concat_type = nn.Linear(dec_dim + dec_emb_dims[0], dec_dim)
        
        # individual outputs
        self.proj = nn.ModuleList([nn.Linear(dec_dim, v) for i, v in dec_vocab_sizes])
        
        # initialise weights
        self.apply(weights_init)

    def forward(self, x, y, attr=None, state=None):
        """
        linear transformer: b x seq_len x dim
        x: compund words (integer encodings); b x s x d x t_type
        attr: sequence of attribute vectors
        y: if training, y is the sequence of target output words 
            (also input to decoder for teacher forcing)
           if inferring, y is the most recent output
        state: decoder state for autoregressive inference
        is_training: training mode (or inference)
        # skips: add skip connections from encoder layers to cross-attention
        # hidden: include final hidden encoder state in all cross attentions
        """
        # embeddings
        enc_emb = torch.cat([e(x[..., i]) for i, e in enumerate(self.enc_embeddings)], 
                            dim=-1)
        enc_emb = self.enc_emb_lin(enc_emb)
        enc_emb = self.enc_pos(enc_emb)
        
        if attr is not None:
            attr_emb = torch.cat([e(attr[..., i]) for i, e in 
                                  enumerate(self.attr_embeddings)], dim=-1)        
        
        dec_emb = torch.cat([e(y[..., i]) for i, e in enumerate(self.dec_embeddings)], 
                            dim=-1)
        dec_emb = self.dec_emb_lin(dec_emb)
        dec_emb = self.dec_pos(dec_emb)
        
        # encoder
        h, skips = self.enc_block(enc_emb)
        
        # skip connections (optional)
        zs = skips if self.skips else [torch.zeros_like(s) for s in skips]
        
        # cross-attend to final encoder hidden state (optional)
        if self.hidden:
            zs = [z + h for z in zs]
        
        # concatenate attributes to latent space (optional)
        if attr is not None:
            if self.attr_pos_emb:
                attr_emb = self.attr_pos(attr_emb)
            zs = torch.cat([zs, attr_emb], dim=-1)
        
        # decoder
        if self.is_training:
            out = self.dec_block(dec_emb, zs)
        else:
            dec_emb = dec_emb.squeeze(0)
            out, state = self.dec_block(dec_emb, zs, state)
        
        # type residual (optional)
        if self.out_t_types[0] == 'type':
            # get type prediction before residual connection
            y_type = self.proj[0](out)

            # concatenate type residual
            type_res = self.dec_embeddings(y[..., 0])
            out = torch.cat([out, type_res], dim=-1)
            out = self.res_concat_type(out)
        
            # project word to tokens
            tokens_out = [y_type] + [p(out) for p in self.proj[1:]]
        
        else:
            tokens_out = [p(out) for p in self.proj]

        # tokens out dims: (batch, seq_len, voc_size)

        return tokens_out, state

    def compute_loss(self, pred_tokens, targets):
        """
        pred_tokens: [(b, seq_len, vocab_size)] (length: t_types)
        targets: (b, s, t_types)
        """
        (b, s, t) = targets.size()
        assert len(pred_tokens) == t
        assert (b, s) == pred_tokens[0].size[0: 2]
        assert b == 1

        criterion = nn.CrossEntropyLoss()
        losses = []

        for idx in range(len(self.out_t_types)):
            pred = pred_tokens[idx].permute(0, 2, 1)
            losses.append(criterion(pred, targets[..., idx]))
            
        overall_loss = torch.stack(losses).sum()
        
        return overall_loss, losses    
    
    def infer(self, tokens_out):
        """
        convert token logits to word
        """
        return torch.cat([torch.argmax(t, 2, keepdim=True) for t in tokens_out], dim=-1)

