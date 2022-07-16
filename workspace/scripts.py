"""
train and evaluate functions
"""
import os

import pickle

from torch.utils.data import dataloader_experimental

from model import models
import data_utils


def train(data,
          epochs, 
          validation_ratio=
          max_grad_norm=3):

    # load metadata
    with open(os.path.join(data_base, 'metadata.pkl'), 'rb') as f:
        meta = pickle.load(f)

    # get data


def evaluate():
    ...


class Trainer():
    def __init__(self, 
                 path,
                 data_base,
                 val_ratio,
                 test_ratio,
                 seed=1,
                 split_mode='random'):
        self.path = path

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
        
        # create folders
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, 'runs'), exist_ok=True)
        
        # save object
        self.class_save()
    
    def class_save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self, f)


# class Tester():
#     def __init__(self, 
#                  path,
#                  data_base):
#         self.path = path
