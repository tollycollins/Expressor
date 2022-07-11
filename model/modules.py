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
from fast_transformers.transformers import TransformerEncoderLayer, TransformerDecoderLayer
from fast_transformers.masking import FullMask, LengthMask, TriangularCausalMask
from fast_transformers.events import EventDispatcher


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim_emb):
        super().__init__()
        
        self.dim_model = dim_emb
        self.emb = nn.Embedding(vocab_size, dim_emb)
        
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.dim_emb)


class PositionalEncoding(nn.module):
    def __init__(self, dim_model, dropout=0.1, max_len=20000):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        pos_enc = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * \
                             (-math.log(10000.0) / dim_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc)
        
    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1), :]
        return self.dropout(x)


class EncoderBlock(nn.module):
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
        length_mask = LengthMask(x.new_full((N, ), L, dtype=torch.int64))

        # apply layers
        skips = []
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask)
            skips.append(x)
        
        # optional normalization
        if self.norm is not None:
            x = self.norm(x)
        
        return x, skips
    

class DecoderBlock(nn.module):
    def __init__(self, layers):
        super().__init__()
        self.layers=nn.ModuleList(layers)
        self.event_dispatcher = EventDispatcher.get()
    
    def forward(self, x, zs):
        # masks
        N = x.shape[0]
        L = x.shape[1]
        L_prime = zs[0].shape[1]
        x_mask = TriangularCausalMask(L, device=x.device)
        x_length_mask = LengthMask(x.new_full((N, ), L, dtype=torch.int64))
        zs_mask = FullMask(L, L_prime, device=x.device)
        zs_length_mask = LengthMask(x.new_full((N, ), L_prime, dtype=torch.int64))

        # apply layers
        for idx, layer in enumerate(self.layers):
            x = layer(x, zs[idx], x_mask=x_mask, x_length_mask=x_length_mask, 
                      memory_mask=zs_mask, memory_length_mask=zs_length_mask)
        
        return x
        
        
        