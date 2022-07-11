import torch
import torch.nn as nn
from fast_transformers.transformers import TransformerEncoderLayer, TransformerDecoderLayer


class Expressor(nn.Module):
    def __init__(self):
        super().__init__()
        