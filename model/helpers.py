"""
Transformer utilities

References:
    https://github.com/YatingMusic/MuseMorphose/blob/main/model/transformer_helpers.py
"""

import numpy as np
from torch import nn

def weight_init_normal(weight, normal_std):
  nn.init.normal_(weight, 0.0, normal_std)

def weight_init_orthogonal(weight, gain):
  nn.init.orthogonal_(weight, gain)

def bias_init(bias):
  nn.init.constant_(bias, 0.0)
  
def weights_init(m, verbose=False):
    classname = m.__class__.__name__

    if verbose:
        print (f'[{classname}] initializing ...')

    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            weight_init_normal(m.weight, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            weight_init_normal(m.weight, 0.01)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            bias_init(m.bias)
    else:
      print (f'[{classname}] not initialized !!')


def network_params(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

