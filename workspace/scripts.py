"""
train and evaluate functions
"""
import os

import pickle

from torch.utils.data import DataLoader

from model import models
import data_utils


class SaverAgent():
    ...


class Controller():

    def __init__(self, 
                 path,
                 data_base,
                 val_ratio,
                 test_ratio,
                 seed=1,
                 split_mode='random'):
        
        # paths
        self.path = path
        self.data_base = data_base
        
        # data split parameters
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.split_mode = split_mode

        # load metadata of full words dataset
        with open(os.path.join(data_base, 'metadata.pkl'), 'rb') as f:
            self.val2idx, self.idx2val, self.meta = pickle.load(f)
        
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
        print("Trainer object saved")


    def train(self,
              epochs,
              batch_size=1,
              in_types=[],
              attr_types=[],
              out_types=[],
              model_args=[],
              model_kwargs={},
              param_path=None,
              max_grad_norm=3):

        # get Dataloaders
        train_loader = DataLoader(data_utils.WordDataset(self.data_base, 
                                                         self.train_ind,
                                                         in_types, attr_types, out_types), 
                                  batch_size=batch_size,
                                  pin_memory=True, 
                                  shuffle=True)
        val_loader = DataLoader(data_utils.WordDataset(self.data_base, 
                                                       self.val_ind,
                                                       in_types, attr_types, out_types),
                                batch_size=batch_size,
                                pin_memory=True, 
                                shuffle=True)
        
        # get vocabulary sizes
        in_pos, attr_pos, out_pos = self.new_positions((in_types, attr_types, out_types))

        # get vocabulary sizes
        in_vocab_sizes = [len(self.idx2val[t]) for t in in_pos.values()]
        attr_vocab_sizes = [len(self.idx2val[t]) for t in attr_pos.values()]
        out_vocab_sizes = [len(self.idx2val[t]) for t in out_pos.values()]
    
        # initialise model
        
        # load model parameters

        # optimiser

        # scheduler

        # saver agent

        # training loop
    

    
    
    def new_positions(self, t_types_tup):
        """
        Get dictionary of {index in word: token type}
        """
        out = []
        for t_types, pos in zip(t_types_tup, ('in_pos', 'attr_pos', 'out_pos')):
            # order old_types by position in word
            old_types = sorted(self.meta[pos].keys(), key=lambda x: self.meta[pos][x])

            # only keep types in new word, and give index as key
            out.append({i: t for i, t in 
                        enumerate([ot for ot in old_types if ot in t_types])})
        
        return out

    


# class Tester():
#     def __init__(self, 
#                  path,
#                  data_base):
#         self.path = path
