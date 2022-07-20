"""
train and evaluate functions

References:
    https://github.com/YatingMusic/compound-word-transformer/blob/main/workspace/uncond/cp-linear/main-cp.py
    https://github.com/YatingMusic/MuseMorphose/blob/main/model/musemorphose.py
    https://github.com/YatingMusic/compound-word-transformer/blob/main/workspace/uncond/cp-linear/saver.py
    https://www.jeremyjordan.me/nn-learning-rate/

 file system linked to Controller object:
    Saves
        Words_0
            Words
                word objects
            controller_save
            Run_000_[name]
                saved_models (by condition)
                saved optimizer
                txt file training log
                save parameters as json
                Tests
                    test outputs
                    test logs
                Renders
                    outputs
                        midi renders of test outputs
                    targets
                        midi renders fo targets from available tokens
            Run_001_[name]
                ...
"""
import os
import time
import logging
import pickle
import json
import math

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, CosineAnnealingLR, LambdaLR,
)
from torch.optim.swa_utils import AveragedModel, SWALR

from fast_transformers.utils import make_mirror

from model.models import Expressor
from model.helpers import network_params
import data_utils


class SaverAgent():
    """
    Handles saving of training data
    Handles model saving
    Creates a new directory for each training run
    
    file modes
    'a': 
        Opens a file for appending. The file pointer is at the end of the file if the file exists. 
        That is, the file is in the append mode. If the file does not exist, it creates a new file for writing.
    'w':
        Opens a file for writing only. Overwrites the file if the file exists.
        If the file does not exist, creates a new file for writing.
    """
    def init(self, save_root,
                   save_name=None, 
                   exp_dir=None,
                   filemode='w'):
        
        self.init_time = time.time()
        self.global_step = 0
        
        if save_name[0:4] != 'run_':
            # directory name
            name = 'run_000'
            names = sorted(filter(lambda x: os.path.isdir(os.path.join(save_root, x)), 
                                    os.listdir(save_root)))
            if names:
                name = f"run_{int(names[-1][4:7]) + 1:03}"
            if save_name:
                name += f"_{save_name}"
            
            # make directory
            os.makedirs(os.path.join(save_root, name), exist_ok=True)
            save_name = name
        
        self.save_dir = os.path.join(save_root, save_name)
        
        # allow for different log directory
        self.exp_dir = self.save_dir if not exp_dir else os.path.join(self.save_dir, exp_dir)
    
        # logging config
        path_logger = os.path.join(self.exp_dir, 'log.txt')
        logging.basicConfig(
                level=logging.DEBUG,
                format='%(message)s',
                filename=path_logger,
                filemode=filemode)
        self.logger = logging.getLogger('training monitor')
        
        # for checking saving
        self.best_val = None
        
        # for early stopping
        self.improvement_ctr = 0
        
    def add_summary_msg(self, msg):
        self.logger.debug(msg)

    def add_summary(self, key, 
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
        
    def save_model(self, model, 
                         optimizer=None, 
                         outdir=None, 
                         name='model'):
        
        if outdir is None:
            outdir = self.save_dir
        print(f' [*] saving model to {outdir}, name: {name}')
        torch.save(model, os.path.join(outdir, name + '.pt'))
        torch.save(model.state_dict(), os.path.join(outdir, name + '_params.pt'))

        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(outdir, name + '_opt.pt'))
    
    def check_save(self, metric,
                         better='lower'):
        
        if self.best_val == None:
            self.best_val = metric
            # for early stopping
            self.improvement_ctr = 0
            return False, 0
        
        elif better == 'lower' and metric < self.best_val or \
             better == 'higher' and metric > self.best_val:
            self.best_val = metric
            # for early stopping
            self.improvement_ctr = 0
            return True, 0
        
        # for early stopping
        self.improvement_ctr += 1
        return False, self.improvement_ctr
            
    def load_model(self, path_exp=None, 
                         device='cpu', 
                         name='model.pt'):
        
        if not path_exp:
            path_exp = self.save_dir
        path_pt = os.path.join(path_exp, name)
        print(' [*] restoring model from', path_pt)
        model = torch.load(path_pt, map_location=torch.device(device))
        return model
        
    def global_step_increment(self):
        self.global_step += 1
        
    def save_params(self, **kwargs):
        # make directory
        save_path = os.path.join(self.save_dir, 'training_params.json')
        os.makedirs(save_path, exist_ok=True)
        with open(save_path) as f:
            json.dump(kwargs)
    
    def load_params(self):
        with open(os.path.join(self.save_dir, 'training_params.json')) as f:
            params = json.load(f)
        return params


class Controller():
    """
    Handles training and inference scripts
    Handles hyperparameter searching
    """
    def __init__(self, path,
                       data_base,
                       val_ratio,
                       test_ratio,
                       seed=1,
                       split_mode='random'):
        
        # paths
        self.path = path                    # this object
        self.data_base = data_base          # data directory
        
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
        print(f"Trainer object saved at {self.path}")

    
    def train(self, epochs,
                    batch_size=1,
                    save_name=None,
                    log_mode='w',
                    grad_acc_freq=None,
                    val_freq=5,
                    in_types=[],
                    attr_types=[],
                    out_types=[],
                    model_args=[],
                    model_kwargs={},
                    param_path=None,
                    init_lr=3e-3,
                    min_lr=1e-6,
                    weight_dec=0,
                    max_grad_norm=3,
                    restart_anneal=True,
                    sch_Tmult=1,
                    sch_warm=0.05,
                    swa_start=0.7,
                    swa_init=0.001,
                    n_eval_init=1,
                    save_cond='loss',
                    early_stop=50):
        
        # get Dataloaders
        train_loader = DataLoader(data_utils.WordDataset(self.data_base, self.meta, 
                                                         self.train_ind,
                                                         in_types, attr_types, out_types), 
                                  batch_size=batch_size, pin_memory=True, shuffle=True)
        val_loader = DataLoader(data_utils.WordDataset(self.data_base, self.meta, 
                                                       self.val_ind,
                                                       in_types, attr_types, out_types),
                                batch_size=batch_size, pin_memory=True, shuffle=True)
        
        # get positions of tokens in words
        in_pos, attr_pos, out_pos = self.new_positions((in_types, attr_types, out_types))

        # get vocabulary sizes
        in_vocab_sizes = [len(self.idx2val[t]) for t in in_pos.values()]
        attr_vocab_sizes = [len(self.idx2val[t]) for t in attr_pos.values()]
        out_vocab_sizes = [len(self.idx2val[t]) for t in out_pos.values()]
    
        # initialise model
        model = Expressor(in_types, attr_types, out_types,
                          in_vocab_sizes, attr_vocab_sizes, out_vocab_sizes,
                          *model_args, 
                          is_training=True, 
                          **model_kwargs)
        
        # pair model with a recurrent version for evaluation
        eval_model = Expressor(in_types, attr_types, out_types,
                               in_vocab_sizes, attr_vocab_sizes, out_vocab_sizes,
                               *model_args, 
                               is_training=False, 
                               **model_kwargs)
        make_mirror(model, eval_model)
        
        n_params = network_params(model)
        print(f"Model parameters: {n_params}")
        
        # load model parameters
        if param_path:
            print("Loading model parameters from {param_path}")
            model.load_state_dict(torch.load(param_path))

        # optimiser
        optimizer = Adam(model.parameters(), lr=init_lr, weight_decay=weight_dec)
        
        # schedulers (chained)
        schedulers = []
        # cosine anneal lr decay
        eta_min = math.pow(min_lr / init_lr, 2/3 if restart_anneal else 1)
        schedulers.append(CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3))
        # add restart element to lr decay (optional)
        if restart_anneal:
            schedulers.append(CosineAnnealingWarmRestarts(
                optimizer, sch_warm * epochs, sch_Tmult, eta_min=math.sqrt(eta_min)
            ))
        # add warm-up to lr (optional)
        if sch_warm:
            w_len = sch_warm * epochs
            schedulers.append(LambdaLR(
                optimizer, lambda x: min(1, 0.7 + 0.3 * (1 - (w_len - x) / w_len))
            ))
        
        # stochastic weight averaging
        swa_model = None
        if swa_start:
            swa_scheduler = SWALR(optimizer, swa_lr=swa_init)

        # saver agent
        saver = SaverAgent(self.path, save_name=save_name, filemode=log_mode)
        saver.add_summary_msg(' > # parameters: {n_params}')
        saver.save_params(in_pos=in_pos, attr_pos=attr_pos, out_pos=out_pos,
                          model_args=[in_types, attr_types, out_types, 
                                      in_vocab_sizes, attr_vocab_sizes, out_vocab_sizes, 
                                      *model_args],
                          model_kwargs=model_kwargs)
        
        # handle device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        eval_model.to(device)
        
        # set up gradient accumulation over multiple batches
        if not grad_acc_freq:
            grad_acc_freq = 1        
        
        print("Starting training ...")

        # training loop        
        for epoch in range(epochs):
            
            # record time for logging
            start_time = time.time()
            
            # track cumulative loss
            cum_loss = 0
            cum_losses = np.zeros(len(out_vocab_sizes))
            
            # reset gradients
            optimizer.zero_grad()
        
            for idx, data in enumerate(train_loader, 1):
                
                saver.global_step_increment()
                
                # unpack data
                enc_in = data['in'].to(device)
                attr_in = data['attr'].to(device)
                targets = data['out'].to(device)
                
                # forward pass
                tokens_out, _ = model(enc_in, targets, attr_in, state=None)
                
                # calculate losses
                total_loss, losses = model.compute_loss(tokens_out, targets)
                
                # backward pass
                if (len(train_loader) - idx) >= grad_acc_freq:
                    total_loss /= grad_acc_freq
                total_loss.backward()
                
                # update cumulative loss for epoch
                total_loss.detach_()
                cum_loss += total_loss.item()
                cum_losses += np.array([l.item() for l in losses])
                
                # weights update
                if idx % grad_acc_freq == 0 or (len(train_loader) - idx) < grad_acc_freq:
                    if max_grad_norm is not None:
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # log
                saver.add_summary('batch loss', total_loss.item())
            
            # update learning rate
            if epoch > epochs * swa_start:
                if swa_model is None:
                    swa_model = AveragedModel(model)
                else:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()  
            else:
                for scheduler in schedulers:
                    scheduler.step()
            
            # print
            runtime = time.time() - start_time
            cum_loss /= len(train_loader)
            cum_losses /= len(train_loader)
            print(f"----- epoch: {epoch + 1}/{epochs} | Loss: {cum_loss:08f}" \
                  f"| time: {runtime} -----")
            print('    > ', ' | '.join(f"{t}: {l:06f}" for t, l in zip(out_pos.values(), 
                                                                       cum_losses)))
            
            # log training info
            saver.add_summary('epoch loss', cum_loss)
            for t_type, loss in zip(out_pos.values, cum_losses):
                saver.add_summary(f"{t_type} loss", loss)
            
            # validation
            if (epoch + 1) % val_freq == 0 or epoch + 1 == epochs:
                
                # get time for logging
                eval_start_time = time.time()
                
                # get evaluation metrics
                metrics = self.evaluate(eval_model,
                                        device=device,
                                        val_loader=val_loader,
                                        n_eval_init=n_eval_init)

                # ensure model is in training mode
                model.train()
                
                # print validation info
                runtime = time.time() - eval_start_time
                print(f"*** Validation Loss: {metrics['loss']:08f} | time: {runtime} ***")
                print('    > ', ' | '.join(f"{t}: {l:06f}" for t, l in zip(out_pos.values(), 
                                                                           metrics['losses'])))
                
                # log validation info
                saver.add_summary('validation loss', metrics['loss'])
                for t_type, loss in zip(out_pos.values, metrics['losses']):
                    saver.add_summary(f"validation {t_type} loss", loss)

                # save model
                if save_cond == 'val_loss':
                    save, es_ctr = saver.check_save(metrics['loss'], 'lower')
                
                if save:
                    saver.save_model(model, optimizer)
                
                # early stopping
                if early_stop and es_ctr > early_stop / val_freq:
                    saver.add_summary_msg(f"Early stopping occurred after {epoch + 1} epochs")
                    print("Early stopping occurred")
                    break

        print("Training complete")
            
                
    def evaluate(self, model,
                       device,
                       loader,
                       n_eval_init,
                       output_path=None):
        
        # ensure model is in evaluation mode
        model.eval()
        
        # track cumulative loss
        cum_loss = 0
        cum_losses = np.zeros(len(model.dec_vocab_sizes))
        
        with torch.no_grad():
            for data in loader:
                
                 # unpack data
                enc_in = data['in'].to(device)
                attr_in = data['attr'].to(device)
                targets = data['out'].to(device)
                name = data['name']
                
                # get initial words for autoregressive inference
                y_init = targets[:n_eval_init]
                
                # forward pass
                y_pred = model.infer(enc_in, targets, y_init, attr_in)
                
                # calculate losses
                total_loss, losses = model.compute_loss(y_pred, targets)                               
        
                cum_loss += total_loss
                cum_losses += np.array([l.item() for l in losses])

                # Evaluation metrics
                ...
                
                # save output for testing
                if output_path:
                    filename = name + '_pred_tokens.pkl'
                    with open(filename, 'wb') as f:
                        pickle.dump(y_pred, targets)
        
        # record output info
        output = {}
        output['loss'] = cum_loss / len(loader)
        output['losses'] = cum_losses / len(loader)
        
        return output

    
    def test(self, dir_name,
                   filemode='w',
                   n_tests=1,
                   n_eval_init=1,
                   save_outputs=True):
        
        # get Saver agent
        head, tail = os.path.split(os.path.normpath(dir_name))
        saver = SaverAgent(head, save_name=tail, exp_dir='tests', filemode=filemode)
        
        # load model
        model = saver.load_model()

        # handle device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # get dataloader
        loader = DataLoader(data_utils.WordDataset(self.data_base, self.meta, 
                                                   self.test_ind, model.in_types, 
                                                   model.attr_types, model.out_types,
                                                   max_len=n_tests),
                            batch_size=1, shuffle=False)

        # time for logging
        start_time = time.time()

        # get evaluation metrics
        metrics = self.evaluate(model,
                                device=device,
                                loader=loader,
                                n_eval_init=n_eval_init,
                                output_path=saver.exp_dir if save_outputs else None)

        # print validation info
        runtime = time.time() - start_time
        out_pos = self.new_positions(model.out_types)
        print(f"*** Testing Loss: {metrics['loss']:08f} | time: {runtime} ***")
        print('    > ', ' | '.join(f"{t}: {l:06f}" for t, l in zip(out_pos.values(), 
                                                                metrics['losses'])))
                
        # log validation info
        saver.add_summary(':oss', metrics['loss'])
        for t_type, loss in zip(out_pos.values, metrics['losses']):
            saver.add_summary(f"{t_type} loss", loss)
        
    
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

    
    def hyper_search(self):
        ...


    def render(self):
        ...

