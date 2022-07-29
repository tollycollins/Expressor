"""
Compound Word Linear Transformer Modules
References:
    https://github.com/YatingMusic/compound-word-transformer/blob/main/workspace/uncond/cp-linear/main-cp.py
    https://fast-transformers.github.io/
    https://fast-transformers.github.io/api_docs/fast_transformers/transformers.html
    https://github.com/YatingMusic/MuseMorphose/blob/main/model/musemorphose.py
"""
import math

import torch
import torch.nn as nn
from fast_transformers.masking import FullMask, LengthMask, TriangularCausalMask
from fast_transformers.events import EventDispatcher


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim_emb):
        super().__init__()
        
        self.dim_emb = dim_emb
        self.emb = nn.Embedding(vocab_size, dim_emb)
        
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.dim_emb)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=20000):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        pos_enc = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * \
                             (-math.log(20000.0) / dim_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc)
        
    def forward(self, x, seq_pos=None):
        """
        args:
            x: tensor (b, n, d), or (b, d) in recurrent case
            seq_pos: position of x in output sequence for recurrent version
        """
        # standard encoding
        if len(x.shape) == 3:
            x = self.dropout(x) + self.pos_enc[:, :x.size(1), :]
        # encoding for recurrent version
        elif len(x.shape) == 2:
            x = self.dropout(x) + self.pos_enc[:, seq_pos, :]
        return x


class EncoderBlock(nn.Module):
    def __init__(self, layers, norm_layer=None):
        """
        args:
            layers: TransformerEncoderLayer instances
            norm_layer: Normalization layer applied to final output
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get()

    def forward(self, x):
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = FullMask(L, device=x.device)
        length_mask = LengthMask(x.new_full((N, ), L, dtype=torch.long))
        
        # apply layers
        skips = []
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask)
            skips.append(x)
        
        # optional normalization
        if self.norm is not None:
            x = self.norm(x)
        
        return x, skips
    

class DecoderBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.event_dispatcher = EventDispatcher.get()
    
    def forward(self, y, zs):
        """
        y: for teacher forcing, list of target outputs (batch_len, seq_len, dim_dec)
        zs: list of cross attention keys/values (batch_len, seq_len, dim_attr)
        """
        # masks
        N = y.shape[0]
        L = y.shape[1]
        L_prime = zs[0].shape[1]
        y_mask = TriangularCausalMask(L, device=y.device)
        y_length_mask = LengthMask(y.new_full((N, ), L, dtype=torch.int64))
        zs_mask = FullMask(N=L, M=L_prime, device=y.device)
        zs_length_mask = LengthMask(y.new_full((N, ), L_prime, dtype=torch.int64))

        # apply layers
        for idx, layer in enumerate(self.layers):
            y = layer(y, zs[idx], x_mask=y_mask, 
                      x_length_mask=y_length_mask, 
                      memory_mask=zs_mask, memory_length_mask=zs_length_mask)
        
        return y
        
        
class RecurrentDecoderBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.event_dispatcher = EventDispatcher.get()

    def forward(self, y, zs, state=None):
        """
        y: previous output (batch_len, seq_len, dim_dec)
        zs: list of cross attention keys/values (batch_len, seq_len, dim_attr)
        state: List of objects to be passed to each transformer decoder layers
        """
        # initialise state if not given
        if state is None:
            state = [None] * len(self.layers)
        
        # mask
        N = y.shape[0]

        L_prime = zs[0].shape[1]
        zs_length_mask = LengthMask(y.new_full((N,), L_prime, dtype=torch.int64))
        
        # apply layers
        for idx, layer in enumerate(self.layers):
            y, s = layer(y, zs[idx], memory_length_mask=zs_length_mask, state=state[idx])
            state[idx] = s
        
        return y, state

