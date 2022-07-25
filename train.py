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

from scripts.scripts import Controller


params = {
    'baseline_t1': {
        'init': {
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "seed": 1,
            "split_mode": 'random'
        },
        'test': {
            'kwargs': {
                "batch_size": 1,
                "save_name": 'test',
                "log_mode": 'w',
                "grad_acc_freq": None,
                "val_freq": 1,
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
                "save_cond": 'val_loss',
                "early_stop": 50,
                "max_train_size": 3,
                "max_eval_size": 1,
                "print_model": True
            }
        },
        'baseline': {
            'kwargs': {
                "batch_size": 1,
                "save_name": 'baseline',
                "log_mode": 'w',
                "grad_acc_freq": None,
                "val_freq": 1,
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
                "save_cond": 'val_loss',
                "early_stop": 50,
                "max_train_size": 3,
                "max_eval_size": 1,
                "print_model": False
}
        },
        'baseline_small': {
            'kwargs': {
                "batch_size": 1,
                "save_name": 'baseline',
                "log_mode": 'w',
                "grad_acc_freq": None,
                "val_freq": 8,
                "in_types": ['type', 'beat', 'tempo_band', 'pitch', 'start', 'dur_full', 
                             'dur_fract'],
                "attr_types": [],
                "out_types": ['type', 'ibi', 'local_vel_mean', 'artic_whole', 'artic_fract', 
                              'timing_dev_whole', 'timing_dev_fract', 'note_vel_diff'],
                "model_args": [
                        [4, 32, 8, 64, 64, 32, 64], 256, 8, 4, 1024, 
                        [4, 128, 64, 128, 16, 16, 64, 128], 256, 8, 4, 1024
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
                "save_cond": 'val_loss',
                "early_stop": 50,
                "max_train_size": 3,
                "max_eval_size": 1,
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
    epochs = int(sys.argv[3])
    
    # deal with Google Colab
    save_dir_name = os.path.join('saves', sys.argv[1])
    if os.getcwd() == '/content/Expressor':
        save_dir_name = os.path.join('../gdrive/MyDrive/QMUL/Dissertation', dir_name)
    
    # load controller
    try:
        with open(os.path.join(save_dir_name, 'controller.pkl'), 'rb') as f:
            controller = pickle.load(f)
    except FileNotFoundError:
        controller = Controller(save_dir_name.replace('\\', '/'), 
                                os.path.join(save_dir_name, 'words').replace('\\', '/'), 
                                **params[dir_name]['init'])
        
    
    controller.train(epochs, **params[dir_name][train_name]['kwargs'])
    
