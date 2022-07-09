"""
Compound Word Linear Transformer Modules
"""
import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim_model):
        super().__init__()
        
        self.dim_model = dim_model
        self.emb = nn.Embedding(vocab_size, dim_model)
        
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.dim_model)


class PositionalEncoding(nn.module):
    def __init__(self, dim_model, dropout=0.1, max_len=20000):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        pos_enc = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc)
        
    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1), :]
        return self.dropout(x)


class EncoderBlock(nn.module):
    def __init__(self):
        super().__init__()
        
        
    
    
        
        
        