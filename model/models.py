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
from fast_transformers.recurrent.attention import (
    RecurrentAttentionLayer, RecurrentCrossAttentionLayer, 
    RecurrentLinearAttention, RecurrentCrossLinearAttention
)
from fast_transformers.attention import AttentionLayer, LinearAttention, CausalLinearAttention

from modules import (
    Embedding, PositionalEncoding, EncoderBlock, DecoderBlock, RecurrentDecoderBlock
)
from helpers import weights_init


class Expressor(nn.Module):
    """
    Parallel encoder-decoder blocks with optional skip connections.  
    Generation-guiding tokens can be added to cross-attention latent space.  
    """
    def __init__(self, 
                 in_t_types, attr_t_types, out_t_types,
                 enc_vocab_sizes, attr_vocab_sizes, dec_vocab_sizes,
                 enc_emb_dims, enc_dim, enc_n_layers, enc_heads, enc_ff_dim,
                 dec_emb_dims, dec_dim, dec_n_layers, dec_heads, dec_ff_dim,
                 attr_emb_dims=[],
                 attr_pos_emb=False,
                 enc_norm_layer=True,
                 enc_dropout=0.1, enc_act='relu',
                 dec_dropout=0.1, dec_act='relu',
                 skips=True,
                 hidden=True,
                 init_verbose=False,
                 is_training=True):
        super().__init__()

        # record hyperparameters
        self.in_t_types = in_t_types
        self.attr_t_types = attr_t_types
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
        if len(attr_t_types):
            self.attr_embeddings = nn.ModuleList([
                Embedding(v, e) for v, e in zip(attr_vocab_sizes, attr_emb_dims)
            ])
        self.dec_embeddings = nn.ModuleList([
            Embedding(v, e) for v, e in zip(dec_vocab_sizes, dec_emb_dims)
        ])
        
        # linear for embeddings
        self.enc_emb_lin = nn.Linear(np.sum(enc_emb_dims), enc_dim)
        self.dec_emb_lin = nn.Linear(np.sum(dec_emb_dims), dec_dim)
        if enc_dim + (np.sum(attr_emb_dims) or 0) != dec_dim:
            self.latent_lin = nn.Linear(enc_dim + (np.sum(attr_emb_dims) or 0), dec_dim)

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
                    AttentionLayer(CausalLinearAttention(dec_dim), dec_dim, dec_heads), 
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
                    RecurrentAttentionLayer(RecurrentLinearAttention(dec_dim), 
                                            dec_dim, dec_heads), 
                    RecurrentCrossAttentionLayer(RecurrentCrossLinearAttention(dec_dim), 
                                                 dec_dim, dec_heads, 
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
        self.proj = nn.ModuleList([nn.Linear(dec_dim, v) for v in dec_vocab_sizes])
        
        # initialise weights
        self.apply(lambda module: weights_init(module, verbose=init_verbose))

    def forward(self, x, y, attr=None, state=None, y_type=None, seq_pos=None):
        """
        linear transformer: b x seq_len x dim
        x: compund words (integer encodings); (b x s x t_type)
        attr: sequence of attribute vectors
        y: if training, y is the sequence of target output words 
            (also input to decoder for teacher forcing) (b x s x t_type)
           if inferring, y is the most recent output (b x t_type)
        state: decoder state for autoregressive inference
        y_type: pass type integer of intended output for inference mode
        seq_pos: position in output sequence for positional encoding in inference mode
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

        dec_emb = self.dec_pos(dec_emb, seq_pos=seq_pos)            
        
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

        # change dimension of latent vectors to match decoder
        if self.enc_dim + np.sum(self.attr_emb_dims) != self.dec_dim:
            for i in range(len(zs)):
                zs[i] = self.latent_lin(zs[i])
        
        # decoder
        if self.is_training:
            out = self.dec_block(dec_emb, zs)
        else:
            out, state = self.dec_block(dec_emb, zs, state=state)
        
        # type residual (optional)
        if self.out_t_types[0] == 'type':
            # get type prediction before residual connection
            y_type_out = self.proj[0](out)
            
            # get type residual
            if self.is_training:
                type_res = self.dec_embeddings[0](y[..., 0]) 
            else:
                type_res = self.dec_embeddings[0](y_type)
            
            # concatenate type residual    
            out = torch.cat([out, type_res], dim=-1)
            out = self.res_concat_type(out)
        
            # project word to tokens
            tokens_out = [y_type_out] + [p(out) for p in self.proj[1:]]
        
        else:
            tokens_out = [p(out) for p in self.proj]
        
        return tokens_out, state

    def compute_loss(self, pred_tokens, targets):
        """
        pred_tokens: [(b, seq_len, vocab_size)] (length: t_types)
        targets: (b, s, t_types)
        """
        (b, s, t) = targets.size()
        assert len(pred_tokens) == t
        assert (b, s) == pred_tokens[0].size()[0: 2], \
            f"Pred tokens: {[*pred_tokens[0].shape[0: 2], len(pred_tokens)]}, Targets: {(b, s, t)}"

        criterion = nn.CrossEntropyLoss()
        losses = []

        for idx in range(len(self.out_t_types)):
            pred = pred_tokens[idx].permute(0, 2, 1)
            losses.append(criterion(pred, targets[..., idx]))
            
        overall_loss = torch.stack(losses).sum() / len(losses)
        
        return overall_loss, losses    
    
    def infer(self, x, target, n_init, attr=None, seq_len=None):
        """
        Infer an output sequence from a given input and attribute

        x: (b, n, d) input sequence of score,
            compund words with and integer value for each token
        target: (b, n, d) target output sequence with aligned token types, 
            compund words with and integer value for each token
        n_init: number of members of target (along n axis) to be used to initialise 
            the autoregressive process
        attr: (b, n, d') attribute tokens to be added to latent space
        seq_len: length of generated sequence (used to cut off very long sequences)
        """
        # initialise recurrent hidden state
        state = None

        # initialise output sequence
        output = [torch.zeros((target.shape[0], target.shape[1] - n_init, d)) for 
                  d in self.dec_vocab_sizes]
        
        # forward passes for initial sequence to initialise hidden states
        for i in range(n_init):
            # get decoder input
            y = target[:, i, :]
            # get type of next output for skip connection
            y_type = target[:, i + 1, 0]
            # forward pass
            h, state = self.forward(x, y, attr, state, y_type, seq_pos=i)
        
        # add h to output (loop over token types)
        for h_tok, out_tok in zip(h, output):
            out_tok[:, 0, :] = h_tok
        # convert logits to ints
        h = self.logits_to_int(h)
        
        # infer rest of output sequence
        for i in range(n_init + 1, seq_len):
            # get type of next output for skip connection
            y_type = target[:, i, 0]
            # forward pass
            h, state = self.forward(x, h, attr, state, y_type, seq_pos=i - 1)
            # add h to output (loop over token types)
            for h_tok, out_tok in zip(h, output):
                out_tok[:, i - n_init, :] = h_tok
            # convert logits to ints
            h = self.logits_to_int(h)
        
        # check sequence length
        assert output[0].shape[1] == x.shape[1] - n_init
        
        return output
    
    @staticmethod
    def logits_to_int(x):
        return torch.stack([torch.argmax(t, -1) for t in x], dim=-1)

