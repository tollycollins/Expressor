"""
split data function
Dataset class
"""
import os
import pickle

import torch
from torch.utils.data import Dataset


def get_split_data(data_path, test_ratio=0.1, validation_ration=0.2,
               max_examples=None, seed=1):
    """
    
    """
    ...


class WordDataset(Dataset):
    def __init__(self, data, t_pos,
                 pitch_aug_range=None):
        """
        data: list(seq(words))
        """
        super().__init__()
        
        self.data = data
        self.t_pos = t_pos
        
        self.length = len(data)
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, index):
            data = self.data[index]
            
            # optional augmentation
            
            return torch.as_tensor(data)
        
        
        
        