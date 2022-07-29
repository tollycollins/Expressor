"""
Model runs

Default training params:
    "batch_size": 1,
    "save_name": None,
    "log_mode": 'w',
    "grad_acc_freq": None,
    "val_freq": 5,
    "in_types": [],
    "attr_types": [],
    "out_types": [],
    "model_args": [],
    "model_kwargs": {},
    "param_path": None,
    "init_lr": 3e-3,
    "min_lr": 1e-6,
    "weight_dec": 0,
    "max_grad_norm": 3,
    "restart_anneal": True,
    "sch_Tmult": 1,
    "sch_warm": 0.05,
    "swa_start": 0.7,
    "swa_init": 0.001,
    "n_eval_init": 1,
    "save_cond": 'loss',
    "early_stop": 50,
    "max_train_size": None,
    "max_eval_size": None,
    "print_model": False
"""
import os
import sys
import pickle

from workspace.scripts import Controller


params = {
    'baseline_t1': {
        'init': {
            "val_ratio": 0.15,
            "test_ratio": 0.1,
            "seed": 1,
            "split_mode": 'random'
        },
        'test': {
            'searches': {
                'def': {
                    'print_model': [True]
                },
                'swa_test': {
                    'val_freq': [3, 3],
                    'model_args': [
                        [[10, 10, 10, 10, 10, 10, 10], 64, 3, 2, 128, 
                         [10, 30, 10, 10, 10, 10, 10, 10], 64, 3, 2, 128],
                        [[4, 4, 4, 4, 4, 4, 4], 8, 2, 1, 16, 
                         [4, 4, 4, 4, 4, 4, 4, 4], 8, 2, 1, 16]
                    ],
                    'max_train_size': [4, 2],
                    'val_freq': [0, 2]
                },
                'sched_test': {
                    'val_freq': [0],
                    'sch_restart_len': [5]
                },
                'batching': {
                    'train_batch_size': [3, 1, 1, 1],
                    'train_seq_len': [None, 200, None, None],
                    'val_batch_size': [3, 3, 1, 1],
                    'val_seq_len': [None, 200, 200, None],
                    'grad_acc_freq': [None, None, None, 4],
                    'val_freq': [2, 2, 2, 2],
                    'max_train_size': [8, 3, 3, 10],
                    'max_eval_size': [3, 4, 1, 1],
                    'val_freq': [2, 2, 2, 0],
                },
                'enc_dec_dims': {
                    'model_args': [
                        [[2, 2, 2, 2, 2, 2, 2], 16, 3, 2, 32, 
                         [2, 2, 2, 2, 2, 2, 2, 2], 32, 3, 2, 64]                 
                    ],
                    'val_freq': [0]
                }
            },
            'kwargs': {
                "train_batch_size": 1,
                "train_seq_len": None,
                "save_name": 'test',
                "log_mode": 'w',
                "grad_acc_freq": None,
                "val_freq": 1,
                "earliest_val": 0,
                "val_batch_size": 1,
                "val_seq_len": None,
                "in_types": ['type', 'beat', 'tempo_band', 'pitch', 'start', 'dur_full', 
                             'dur_fract'],
                "attr_types": [],
                "out_types": ['type', 'ibi', 'local_vel_mean', 'artic_whole', 'artic_fract', 
                              'timing_dev_whole', 'timing_dev_fract', 'note_vel_diff'],
                "model_args": [
                        [10, 10, 10, 10, 10, 10, 10], 64, 3, 2, 128, 
                        [10, 30, 10, 10, 10, 10, 10, 10], 64, 3, 2, 128
                    ],
                "model_kwargs": {
                        "attr_emb_dims": [],
                        "attr_pos_emb": False,
                        "enc_norm_layer": True,
                        "enc_dropout": 0.1,
                        "enc_act": 'relu',
                        "dec_dropout": 0.1,
                        "dec_act": 'relu',
                        "skips": True,
                        "hidden": True,
                        "init_verbose": False
                    },
                "param_path": None,
                "laod_opt": False,
                "init_lr": 3e-3,
                "min_lr": 1e-6,
                "weight_dec": 0,
                "max_grad_norm": 3,
                "restart_anneal": True,
                "sch_warm_time": 0.1,
                "sch_restart_len": 10,
                "sch_restart_proportion": 0.15,
                "sch_warm_factor": 0.5,
                "swa_start": None,
                "swa_init": 0.001,
                "n_eval_init": 1,
                "save_cond": 'val_loss',
                "early_stop": 50,
                "max_train_size": 3,
                "max_eval_size": 1,
                "print_model": False
            }
        },
        'baseline': {
            'kwargs': {
                "train_batch_size": 1,
                "train_seq_len": None,
                "save_name": 'baseline',
                "log_mode": 'w',
                "grad_acc_freq": None,
                "val_freq": 1,
                "earliest_val": 0,
                "val_batch_size": 1,
                "val_seq_len": None,
                "in_types": ['type', 'beat', 'tempo_band', 'pitch', 'start', 'dur_full', 
                             'dur_fract'],
                "attr_types": [],
                "out_types": ['type', 'ibi', 'local_vel_mean', 'artic_whole', 'artic_fract', 
                              'timing_dev_whole', 'timing_dev_fract', 'note_vel_diff'],
                "model_args": [
                        [4, 32, 8, 128, 128, 32, 64], 512, 12, 8, 2048, 
                        [128, 128, 128, 32, 32, 128, 128, 128], 512, 12, 8, 2048
                    ],
                "model_kwargs": {
                        "attr_emb_dims": [],
                        "attr_pos_emb": False,
                        "enc_norm_layer": True,
                        "enc_dropout": 0.1,
                        "enc_act": 'relu',
                        "dec_dropout": 0.1,
                        "dec_act": 'relu',
                        "skips": False,
                        "hidden": True,
                        "init_verbose": False
                    },
                "param_path": None,
                "laod_opt": False,
                "init_lr": 3e-3,
                "min_lr": 1e-6,
                "weight_dec": 0,
                "max_grad_norm": 3,
                "restart_anneal": True,
                "sch_warm_time": 0.1,
                "sch_restart_len": 10,
                "sch_restart_proportion": 0.15,
                "sch_warm_factor": 0.5,
                "swa_start": None,
                "swa_init": 0.001,
                "n_eval_init": 1,
                "save_cond": 'val_loss',
                "early_stop": 50,
                "max_train_size": 3,
                "max_eval_size": 1,
                "print_model": False
            }
        },
        'small': {
            'searches': {
                'def': {},
                'lrs': {
                  'init_lr': [1e-2, 3e-3, 1e-3, 3e-4],
                  'min_lr': [1e-6, 1e-6, 1e-6, 1e-7],
                  'val_freq': [0, 0, 0, 16],
                  'earliest_val': [72, 72, 72, 72],
                  'max_train_size': [128, 128, 128, 128],
                  'train_batch_size': [8, 8, 8, 8],
                  'train_seq_len': [4000, 4000, 4000, 4000],
                  'val_seq_len': [200, 200, 200, 1000],
                  'grad_acc_freq': [None, None, None, None],
                  'val_batch_size': [8, 8, 8, 8],
                  'max_eval_size': [1, 1, 1, 16],
                  'save_name': ['1e-2', '3e-3', '1e-3', '3e-4']
                 }
            },
            'kwargs': {
                "train_batch_size": 1,
                "train_seq_len": None,
                "save_name": 'small',
                "log_mode": 'w',
                "grad_acc_freq": None,
                "val_freq": 8,
                "earliest_val": 0,
                "val_batch_size": 1,
                "val_seq_len": None,
                "in_types": ['type', 'beat', 'tempo_band', 'pitch', 'start', 'dur_full', 
                             'dur_fract'],
                "attr_types": [],
                "out_types": ['type', 'ibi', 'local_vel_mean', 'artic', 
                              'timing_dev', 'note_vel_diff'],
                "model_args": [
                        [3, 4, 2, 2, 8, 4, 4], 64, 6, 4, 256, 
                        [3, 4, 4, 8, 8, 8], 64, 6, 4, 256
                    ],
                "model_kwargs": {
                        "attr_emb_dims": [],
                        "attr_pos_emb": False,
                        "enc_norm_layer": True,
                        "enc_dropout": 0.1,
                        "enc_act": 'relu',
                        "dec_dropout": 0.1,
                        "dec_act": 'relu',
                        "skips": False,
                        "hidden": True,
                        "init_verbose": False
                    },
                "param_path": None,
                "laod_opt": False,
                "init_lr": 3e-3,
                "min_lr": 1e-6,
                "weight_dec": 0,
                "max_grad_norm": 3,
                "restart_anneal": True,
                "sch_warm_time": 0.1,
                "sch_restart_len": 10,
                "sch_restart_proportion": 0.15,
                "sch_warm_factor": 0.5,
                "swa_start": None,
                "swa_init": 0.001,
                "n_eval_init": 1,
                "save_cond": 'val_loss',
                "early_stop": 50,
                "max_train_size": None,
                "max_eval_size": 32,
                "print_model": False
            }
        }
    }
}


if __name__ == '__main__':
    """
    args:
        [1]: name of folder for controller object (linked to words sequence) - 
                should match a key in params
        [2]: name of folder for training instance - 
                should match an option for argv[1].values().keys()
        [3]: number of epochs for training
    """
    
    dir_name = sys.argv[1]
    train_name = sys.argv[2]
    search_name = sys.argv[3]
    epochs = int(sys.argv[4])
    try:
        search_type = sys.argv[5]
    except IndexError:
        search_type = 'zip'
    
    # deal with Google Colab
    save_dir_name = os.path.join('saves', dir_name)
    if os.getcwd() == '/content/Expressor':
        save_dir_name = os.path.join('../gdrive/MyDrive/QMUL/Dissertation', save_dir_name)
    
    # load controller
    try:
        with open(os.path.join(save_dir_name, 'controller.pkl'), 'rb') as f:
            controller = pickle.load(f)
    except FileNotFoundError:
        controller = Controller(save_dir_name.replace('\\', '/'), 
                                os.path.join(save_dir_name, 'words').replace('\\', '/'), 
                                **params[dir_name]['init'])
    
    # train
    kwargs = params[dir_name][train_name]['kwargs']
    kwargs['save_name'] = train_name + '_' + search_name
    changes = params[dir_name][train_name]['searches'][search_name]

    controller.hyper_search(kwargs, changes, epochs, search_type)
    
