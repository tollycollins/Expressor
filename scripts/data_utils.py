"""
split data function
Dataset class
"""
import os
import pickle

import torch
from torch.utils.data import Dataset, random_split


def make_random_split(length, ratio, seed=1):
    train_size = int((1 - ratio) * length)
    test_size = length - train_size   

    train_ind, test_ind = random_split(range(length), [train_size, test_size], 
                                       generator=torch.Generator().manual_seed(seed))
    
    return train_ind, test_ind


def get_split_data(names, test_ratio=0.1, validation_ratio=0.2,
                   max_examples=None, seed=1,
                   split_mode="random"):
    """
    Split data into training, validation and test sets

    data: (in, attr, out) - [seq(word)]
    """
    length = len(names)
    
    if max_examples:
        length = min(length, max_examples)   
    
    val = None
    if split_mode == 'random':
        train, test = make_random_split(length, test_ratio, seed)

        if validation_ratio:
            train_ind, val_ind = make_random_split(len(train), validation_ratio, seed)
            train = [train[i] for i in train_ind]
            val = [train[i] for i in val_ind]
    
    return train, val, test


class WordDataset(Dataset):
    def __init__(self, 
                 data_base, 
                 meta,
                 t_idxs,
                 in_types,
                 attr_types,
                 out_types,
                 max_len=None,
                 pitch_aug_range=None):
        """
        data: (in, attr, out) - [seq(words)]
        """
        super().__init__()
        
        with open(os.path.join(data_base, 'words.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        # filter out desired tokens for words, and filter out only desired tracks
        in_pos = [meta['in_pos'][t] for t in in_types].sort()
        self.in_data = [[[word[i] for i in in_pos] for word in track] for \
                        idx, track in enumerate(data[0]) if idx in t_idxs]

        attr_pos = [meta['attr_pos'][t] for t in attr_types].sort()
        self.attr_data = [[[word[i] for i in attr_pos] for word in track] for \
                        idx, track in enumerate(data[1]) if idx in t_idxs]

        out_pos = [meta['out_pos'][t] for t in out_types].sort()
        self.out_data = [[[word[i] for i in out_pos] for word in track] for \
                        idx, track in enumerate(data[2]) if idx in t_idxs]

        self.names = [meta['words_info']['names'][i] for i in t_idxs]
        
        self.length = len(self.in_data)
        if max_len:
            self.length = min(self.length, max_len)
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, index):
            data = {
                'in': torch.as_tensor(self.in_data[index]),
                'attr': torch.as_tensor(self.attr_data[index]) if self.attr_data[index] else None,
                'out': torch.as_tensor(self.out_data[index]),
                'name': self.names[index]
            }
            
            # optional augmentation
            
            return data
        
        