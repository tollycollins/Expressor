"""
split data function
Dataset class
"""
import os
import compress_pickle

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
            val = [train[i] for i in val_ind]
            train = [train[i] for i in train_ind]
    
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
        
        with open(os.path.join(data_base, 'words.xz').replace('\\', '/'), 'rb') as f:
            data = compress_pickle.load(f)
        
        # filter out desired tokens for words, and filter out only desired tracks      
        in_pos = sorted([meta['in_pos'][t] for t in in_types])
        # if 'meta' in meta['in_pos']:
        #     in_pos = [0] + in_pos
        self.in_data = [[[word[i] for i in in_pos] for word in track] for \
                        idx, track in enumerate(data[0]) if idx in t_idxs]
        
        self.attr_data = None
        if len(attr_types):
            attr_pos = sorted([meta['attr_pos'][t] for t in attr_types])
            # if 'meta' in meta['attr_pos']:
            #     attr_pos = [0] + attr_pos
            self.attr_data = [[[word[i] for i in attr_pos] for word in track] for \
                            idx, track in enumerate(data[1]) if idx in t_idxs]

        out_pos = sorted([meta['out_pos'][t] for t in out_types])
        # if 'meta' in meta['out_pos']:
        #     out_pos = [0] + out_pos
        self.out_data = [[[word[i] for i in out_pos] for word in track] for \
                         idx, track in enumerate(data[2]) if idx in t_idxs]

        self.names = [meta['words_info']['names'][i] for i in t_idxs]
        
        self.length = len(self.in_data)
        if max_len:
            self.length = min(self.length, max_len)
            
        print(f"Dataset length: {self.length}")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = {
            'in': torch.as_tensor(self.in_data[index]),
            'out': torch.as_tensor(self.out_data[index]),
            'name': self.names[index]
        }

        if self.attr_data is not None:
            data['attr'] = torch.as_tensor(self.attr_data[index])
        
        # optional augmentation
        
        return data
        
        