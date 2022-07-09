"""
Token-related metrics
"""
import os
import pickle
import collections
import matplotlib.pyplot as plt
import sys
import json
import pandas as pd
import numpy as np


def get_val_frequencies(tokens_root, t_type, verbose=True):
    """
    returns:
        unique_vals: dictionary(set) of {'token_type': {unique values across all tokens}}
    """
    unique_vals = collections.OrderedDict()
    for entry in os.scandir(tokens_root):
        if os.path.isfile(entry):
            if verbose:
                print(f"Listing unique {t_type} tokens for {entry} ...                     ", end='\r')
            with open(entry, 'rb') as f:
                token_data = pickle.load(f)
            if type(token_data[t_type]) is list:
                for token in token_data[t_type]:
                    if token[1] in unique_vals:
                        unique_vals[token[1]] += 1
                    else:
                        unique_vals[token[1]] = 0
            else:
                for tokens in token_data[t_type].values():
                    for token in tokens:
                        if token[1] in unique_vals:
                            unique_vals[token[1]] += 1
                        else:
                            unique_vals[token[1]] = 0
    
    if verbose:
        print("Search complete")
    
    return unique_vals


def histogram(data, save_path, name):
    """
    """
    # prepare data
    df = pd.DataFrame(list(data.items()), columns=[name, 'frequency'])
    
    fig, ax = plt.subplots(figsize=(9, 6))
    # plt.figure(dpi=100)
    # plt.bar(list(data.keys()), data.values())
    # plt.bar(np.arange(len(df)), df['frequency'])
    # plt.title(f"Token frequencies for {name}")
    # plt.xlabel(f"name")
    # plt.ylabel('frequency')
    ax.set_title('Token values')
    ax.bar(np.arange(len(df)), df['frequency'])
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df[name])

    # save bar chart
    save_path = os.path.join(save_path, name)
    plt.savefig(save_path)
    plt.close()
    

def words_lens(words_meta_path):
    """
    returns:
        seq_lens: dict of {name: seq_len}
        max_len: length of longest words sequence in dataset
    """
    seq_lens = dict()
    
    with open(words_meta_path, 'wb') as f:
        data = pickle.load(f, 'rb')
    
    max_len = 0
    for name, v in data[2]['words_info'].items():
            length = v['num_words']
            seq_lens[name] = length
            if length > max_len:
                max_len = length
    
    return seq_lens, max_len



if __name__ == '__main__':
    # tokens_path = sys.argv[1]
    tokens_path = "dataset/tokens/t1"
    
    
    save_path = os.path.join(tokens_path, 'visualisations')
    os.makedirs(save_path, exist_ok=True)
    
    # meta_path = os.path.join(tokens_path, '/metadata')
    # with open(meta_path) as f:
    #     tokens_meta = json.load(f)
    
    # tokens_meta = ['bar', 'beat', 'ibi', 'tempo_band', 'local_tempo', 'time_sig', 
    #                'pitch', 'start', 'duration', 'dur_full', 'dur_fract', 
    #                'local_vel_band', 'local_vel_mean', 'local_vel_std', 'note_vel',
    #                 'note_vel_band', 'note_rel_vel', 'articulation', 'timing_dev', 
    #                 'keys', 'harmonic_quality']
    
    tokens_meta = ['local_vel_mean', 'local_vel_std', 'note_vel',
                   'note_vel_band', 'note_rel_vel', 'articulation', 'timing_dev', 
                   'keys', 'harmonic_quality']
    
    for t_type in tokens_meta:
        unique_vals = get_val_frequencies(tokens_path, t_type=t_type)
        histogram(unique_vals, save_path, t_type)

