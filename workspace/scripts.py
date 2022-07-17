"""
train and evaluate functions

References:
    https://github.com/YatingMusic/compound-word-transformer/blob/main/workspace/uncond/cp-linear/main-cp.py
    https://github.com/YatingMusic/MuseMorphose/blob/main/model/musemorphose.py
    https://github.com/YatingMusic/compound-word-transformer/blob/main/workspace/uncond/cp-linear/saver.py
"""
import os
import time
import logging
import pickle
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from model.models import Expressor
from model.helpers import network_params
import data_utils


class SaverAgent():
    """
    Handles saving of training data
    Creates a new directory for each training run
    
    file modes
    'a': 
        Opens a file for appending. The file pointer is at the end of the file if the file exists. 
        That is, the file is in the append mode. If the file does not exist, it creates a new file for writing.
    'w':
        Opens a file for writing only. Overwrites the file if the file exists.
        If the file does not exist, creates a new file for writing.
    """
    def init(self, 
             save_root,
             name=None, 
             mode='w'):
        
        self.init_time = time.time()
        self.global_step = 0
        
        # directory name
        save_name = 'run_000'
        names = sorted(filter(lambda x: os.path.isdir(os.path.join(save_root, x)), 
                                  os.listdir(save_root)))
        if names:
            save_name = f"run_{int(names[-1][4:7]) + 1:03}"
        if name:
            save_name += f"_{name}"
        
        self.exp_dir = os.path.join(save_root, save_name)
        
        # make directory
        os.makedirs(os.path.join(save_root, self.name), exist_ok=True)
        
        # logging config
        path_logger = os.path.join(self.exp_dir, 'log.txt')
        logging.basicConfig(
                level=logging.DEBUG,
                format='%(message)s',
                filename=path_logger,
                filemode=mode)
        self.logger = logging.getLogger('training monitor')
        
    def add_summary_msg(self, msg):
        self.logger.debug(msg)

    def add_summary(self, 
                    key, 
                    val, 
                    step=None, 
                    cur_time=None):

        if cur_time is None:
            cur_time = time.time() - self.init_time
        if step is None:
            step = self.global_step

        # write msg (key, val, step, time)
        if isinstance(val, float):
            msg_str = f'{key:10s} | {val:.10f} | {step:10d} | {cur_time}'   
        else:
            msg_str = f'{key:10s} | {val} | {step:10d} | {cur_time}'

        self.logger.debug(msg_str)        
        
    def save_model(self, 
                   model, 
                   optimizer=None, 
                   outdir=None, 
                   name='model'):

        if outdir is None:
            outdir = self.exp_dir
        print(f' [*] saving model to {outdir}, name: {name}')
        torch.save(model, os.path.join(outdir, name+'.pt'))
        torch.save(model.state_dict(), os.path.join(outdir, name+'_params.pt'))

        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(outdir, name+'_opt.pt'))
            
    def load_model(self, 
                   path_exp=None, 
                   device='cpu', 
                   name='model.pt'):

        if not path_exp:
            path_exp = self.exp_dir
        path_pt = os.path.join(path_exp, name)
        print(' [*] restoring model from', path_pt)
        model = torch.load(path_pt, map_location=torch.device(device))
        return model
        
    def global_step_increment(self):
        self.global_step += 1
        
    def save_params(self, 
                    **kwargs):
        # make directory
        save_path = os.path.join(self.exp_dir, 'training_params.json')
        os.makedirs(save_path, exist_ok=True)
        with open(save_path) as f:
            json.dump(kwargs)
        


class Controller():
    """
    Handles training and inference scripts
    Handles hyperparameter searching
    """
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
              save_name=None,
              log_mode='w',
              in_types=[],
              attr_types=[],
              out_types=[],
              model_args=[],
              model_kwargs={},
              param_path=None,
              init_lr=0.0001,
              max_grad_norm=3):

        # get Dataloaders
        train_loader = DataLoader(data_utils.WordDataset(self.data_base, self.train_ind,
                                                         in_types, attr_types, out_types), 
                                  batch_size=batch_size, pin_memory=True, shuffle=True)
        val_loader = DataLoader(data_utils.WordDataset(self.data_base, self.val_ind,
                                                       in_types, attr_types, out_types),
                                batch_size=batch_size, pin_memory=True, shuffle=True)
        
        # get vocabulary sizes
        in_pos, attr_pos, out_pos = self.new_positions((in_types, attr_types, out_types))

        # get vocabulary sizes
        in_vocab_sizes = [len(self.idx2val[t]) for t in in_pos.values()]
        attr_vocab_sizes = [len(self.idx2val[t]) for t in attr_pos.values()]
        out_vocab_sizes = [len(self.idx2val[t]) for t in out_pos.values()]
    
        # initialise model
        model = Expressor(*model_args, **model_kwargs)
        model.train()
        
        n_params = network_params(model)
        print(f"Model parameters: {n_params}")
        
        # load model parameters
        if param_path:
            print("Loading model parameters from {param_path}")
            model.load_state_dict(torch.load(param_path))

        # optimiser
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

        # saver agent
        saver = SaverAgent(self.path, name=save_name, mode=log_mode)
        saver.add_summary_msg(' > # parameters: {n_params}')
        
        # handle device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        print("Starting training ...")

        # training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            # track cumulative loss
            total_loss = 0
            total_losses = np.zeros(7)
        
            for data in train_loader:
                
                saver.global_step_increment()
                
                # unpack data
                enc_in = data['in'].to(device)
                attr_in = data['attr'].to(device)
                dec_in = data['out'].to(device)
                
                # train
                
                loss = ...
                
                # update
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                
    

    
    
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


#  file system:
# Saves
#   Words_0
#       controller_save
#       Run_0_[name]
#           saved_models (by condition)
#           saved optimizer
#           txt file training log
#           save parameters as json
#       Run_1_[name]
