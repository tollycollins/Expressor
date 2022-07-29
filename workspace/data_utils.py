"""
split data function
Dataset class
"""
import os
import compress_pickle
import math

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
                 batch_size=1,
                 max_len=None,
                 seq_len=None,
                 pitch_aug_range=None):
        """
        data: (in, attr, out) - [seq(words)]
        """
        super().__init__()
    
        self.max_len = max_len                          # max size of dataset
        # self.pitch_aug_range = pitch_aug_range
        self.batch_size = batch_size
        self.seq_len = seq_len                          # max length of seqeunces in a batch
        
        with open(os.path.join(data_base, 'words.xz').replace('\\', '/'), 'rb') as f:
            data = compress_pickle.load(f)
        
        # filter out desired tokens for words, and filter out only desired tracks      
        in_pos = sorted([meta['in_pos'][t] for t in in_types])
        out_pos = sorted([meta['out_pos'][t] for t in out_types])
        if len(attr_types):
            attr_pos = sorted([meta['attr_pos'][t] for t in attr_types])
        
        if batch_size == 1:
            self.in_data = [torch.as_tensor([[word[i] for i in in_pos] for word in track])
                            for idx, track in enumerate(data[0]) if idx in t_idxs]
        
            self.attr_data = None
            if len(attr_types):
                attr_pos = sorted([meta['attr_pos'][t] for t in attr_types])
                self.attr_data = [torch.as_tensor([[word[i] for i in attr_pos] for word in 
                                  track]) for idx, track in enumerate(data[1]) if idx in 
                                  t_idxs]
            
            self.out_data = [torch.as_tensor([[word[i] for i in out_pos] for word in track]) 
                             for idx, track in enumerate(data[2]) if idx in t_idxs]
            
            self.names = [meta['words_info']['names'][i] for i in t_idxs]
            
            self.length = min(len(self.in_data), max_len or 1100)
        
        # larger batch sizes
        else:
            self.length = min(math.ceil(len(t_idxs) / batch_size), 
                              int(max_len / batch_size) or 1100)

            self.in_data = []
            self.out_data = []
            self.attr_data = [] if len(attr_types) else None
            self.names = []

            for nb in range(self.length):
                idxs = slice(nb * batch_size, min(len(t_idxs), (nb + 1) * batch_size))

                in_d = [torch.as_tensor([[word[i] for i in in_pos] for 
                        word in data[0][idx]]) for idx in t_idxs[idxs]]
                self.in_data.append(self.make_batch(in_d))
            
                out_d = [torch.as_tensor([[word[i] for i in out_pos] for 
                         word in data[2][idx]]) for idx in t_idxs[idxs]]
                self.out_data.append(self.make_batch(out_d))
                
                if len(attr_types):
                    attr_d = [torch.as_tensor([[word[i] for i in attr_pos] for 
                            word in data[1][idx]]) for idx in t_idxs[idxs]]
                    self.attr_data.append(self.make_batch(attr_d))

                self.names.append([meta['words_info']['names'][i] for i in t_idxs[idxs]])
            
        print(f"Dataset length: {self.length}")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        
        data = {
            'in': self.in_data[index],
            'name': self.names[index]
        }       

        if self.attr_data is not None:
            data['attr'] = self.attr_data[index]
        
        if self.seq_len and self.seq_len < data['in'].shape[1]:
            # impose maximum sequence length (sequences can be shorter)
            data['in'] = data['in'][:, :self.seq_len, :]
            data['out'] = self.out_data[index][:, :self.seq_len, :]
            if self.attr_data is not None:
                data['attr'] = self.attr_data[index][:, :self.seq_len, :]
        else:
            data['out'] = self.out_data[index]
            if self.attr_data is not None:
                data['attr'] = self.attr_data[index]
        
        # optional augmentation
        
        return data
    
    def make_batch(self, data):
        
        (s, w) = data[0].shape
        lengths = [d.shape[0] for d in data]
        longest = max(lengths)
        full_len = min(longest, 20000 or self.max_len)
                
        out = torch.zeros((len(data), full_len, w), dtype=torch.long)
        for idx, (dt, length) in enumerate(zip(data, lengths)):
            if length > full_len:
                out[idx, ...] = dt[:full_len, :]
            else:
                out[idx, :length, :] = dt
                
        return out
        
