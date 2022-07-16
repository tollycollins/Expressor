"""
split data function
Dataset class
"""
import os
import pickle

import torch
from torch.utils.data import Dataset, random_split


def random_split(data, length, ratio, seed=1):
    train_size = int((1 - ratio) * length)
    test_size = length - train_size   

    train_ind, test_ind = random_split(range(length), [train_size, test_size], 
                                       generator=torch.Generator().manual_seed(seed))
    
    train_data = [[d[i] for i in train_ind] for d in data]
    test_data = [[d[i] for i in test_ind] for d in data]

    return train_data, test_data


def get_split_data(data_path, test_ratio=0.1, validation_ratio=0.2,
               max_examples=None, seed=1,
               mode="random"):
    """
    Split data into training, validation and test sets

    data: (in, attr, out) - [seq(word)]
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    length = len(data[0])
    
    assert length == len(data[1])
    assert length == len(data[2])
    
    if max_examples:
        length = min(length, max_examples)   
    
    val = None
    if mode == 'random':
        train, test = random_split(data, length, test_ratio, seed)

        if validation_ratio:
            train, val = random_split(train, len(train[0]), validation_ratio, seed)
    
    return train, val, test


class WordDataset(Dataset):
    def __init__(self, data, t_pos,
                 pitch_aug_range=None):
        """
        data: (in, attr, out) - [seq(words)]
        """
        super().__init__()
        
        self.data = data
        self.t_pos = t_pos
        
        self.length = len(data[0])
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, index):
            data = {
                'in': torch.as_tensor(data[0][index]),
                'attr': torch.as_tensor(data[1][index]),
                'out': torch.as_tensor(data[2][index])
            }
            
            # optional augmentation
            
            return data
        
        