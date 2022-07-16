"""
train and evaluate functions
"""
import os

import pickle

from torch.utils.data import Dataloader

from model import models
import data_utils


# def train(data,
#           epochs, 
#           validation_ratio=
#           max_grad_norm=3):
    
#     # load metadata
#     with open(os.path.join(data_base, 'metadata.pkl'), 'rb') as f:
#         meta = pickle.load(f)
    
#     # get data


# def evaluate():
#     ...


class Trainer():
    def __init__(self, 
                 path,
                 data_base,
                 val_ratio,
                 test_ratio,
                 seed=1,
                 split_mode='random'):
        
        # parameters
        self.path = path
        self.data_base = data_base
        
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.split_mode = split_mode

        # load metadata
        with open(os.path.join(data_base, 'metadata.pkl'), 'rb') as f:
            self.meta = pickle.load(f)
        
        # dataset split indices
        self.train_ind, self.val_ind, self.test_ind = \
            data_utils.get_split_data(self.meta['words_info']['names'],
                                      validation_ratio=val_ratio,
                                      test_ratio=test_ratio,
                                      seed=seed,
                                      split_mode=split_mode)
        
        # load data
        with open(os.path.join(data_base, 'words.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        # create Datasets
        train_data = [[d[i] for i in self.train_ind] for d in data]
        val_data = [[d[i] for i in self.val_ind] for d in data]
        test_data = [[d[i] for i in self.test_ind] for d in data]
        
        t_pos = (
            self.meta['in_token_positions'],
            self.meta['attr_token_positions'],
            self.meta['out_token_positions']
        )

        self.train_data = data_utils.WordDataset(train_data, t_pos)
        
        # create folders
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, 'runs'), exist_ok=True)
        
        # save object
        self.class_save()
    
    def class_save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self, f)

    def train(self,
              epochs,
              max_grad_norm=3):

        # load data
        with open(os.path.join(self.data_base, 'words.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        # create Datasets
        train_data = [[d[i] for i in self.train_ind] for d in data]
        val_data = [[d[i] for i in self.val_ind] for d in data]
        
        t_pos = (
            self.meta['in_token_positions'],
            self.meta['attr_token_positions'],
            self.meta['out_token_positions']
        )
        
        train_data = data_utils.WordDataset(train_data, t_pos)
        val_data = data_utils.WordDataset(val_data, t_pos) 

        # get Dataloaders


        

# class Tester():
#     def __init__(self, 
#                  path,
#                  data_base):
#         self.path = path
