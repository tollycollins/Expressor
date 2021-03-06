"""
Token-related metrics
"""
import os
import pickle
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np

import utils

def get_unique_vals(tokens_root, t_type, verbose=True, print_freqs=False,
                        by_track=False):
    """
    returns:
        unique_vals: dictionary(set) of {'token_type': {unique values across all tokens}}
    """
    if by_track:
        all_unique_vals = dict()
    else:
        unique_vals = dict()
    for entry in os.scandir(tokens_root):
        if os.path.isfile(entry):
            if verbose:
                print(f"Listing unique {t_type} tokens for {entry.name} ...                     ", end='\r')
            if by_track:
                unique_vals = dict()
            with open(entry, 'rb') as f:
                token_data = pickle.load(f)
            if type(token_data[t_type]) is list:
                for token in token_data[t_type]:
                    try:
                        t = token[1]
                    except TypeError:
                        t = token
                    if t in unique_vals:
                        unique_vals[t] += 1
                    else:
                        unique_vals[t] = 1
            
            else:
                for tokens in token_data[t_type].values():
                    for token in tokens:
                        try:
                            t = token[1]
                        except IndexError:
                            t = token
                        if t in unique_vals:
                            unique_vals[t] += 1
                        else:
                            unique_vals[t] = 1
            if by_track:
                all_unique_vals[entry.name] = unique_vals
    
    if not by_track:
        all_unique_vals = unique_vals

    if verbose:
        print("\nSearch complete")
    
    if print_freqs:
        if by_track:
            for k, v in sorted(all_unique_vals.items()):
                print(f"{k}: {list(sorted(v.keys()))}")
        else:
            for k, v in sorted(all_unique_vals.items()):
                print(f"{k}: {v}")
    
    return dict(sorted(all_unique_vals.items()))


def vocab_lens(tokens_root, t_types):
    vocab = {}
    for t_type in t_types:
        unique_vals = get_unique_vals(tokens_root, t_type)
        vocab[t_type] = len(unique_vals)
    
    # print
    print("Vocabulary lengths: ")
    for k, v in vocab.items():
        print(f"{k}: {v}")
    
    return vocab


def histogram(data, save_path, name):
    """
    """
    # prepare data
    df = pd.DataFrame(list(data.items()), columns=[name, 'frequency'])
    
    fig, ax = plt.subplots(figsize=(18, 6))
    # plt.figure(dpi=100)
    # plt.bar(list(data.keys()), data.values())
    # plt.bar(np.arange(len(df)), df['frequency'])
    # plt.title(f"Token frequencies for {name}")
    # plt.xlabel(f"name")
    # plt.ylabel('frequency')
    ax.set_title('Token values')
    ax.bar(np.arange(len(df)), df['frequency'])
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df[name], rotation=45, ha='right')

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
    
    with open(words_meta_path, 'rb') as f:
        data = pickle.load(f)
    
    lengths = data[2]['words_info']['lengths']
    for name, length in zip(data[2]['words_info']['names'], lengths):
            seq_lens[name] = length
    
    return seq_lens, sorted(lengths), max(lengths)


if __name__ == '__main__':
    """
    token names: 
        t1
    """
    
    name = sys.argv[1]
    metric  = sys.argv[2]

    tokens_path = os.path.join("dataset/tokens", name)
    words_meta_path = os.path.join("saves", name, "words/metadata.pkl")
    
    
    save_path = os.path.join(tokens_path, 'visualisations')
    os.makedirs(save_path, exist_ok=True)
    
    # tokens_meta = ['bar', 'beat', 'ibi', 'tempo_band', 'local_tempo', 'time_sig', 
    #                'pitch', 'start', 'duration', 'dur_full', 'dur_fract', 
    #                'local_vel_band', 'local_vel_mean', 'local_vel_std', 'note_vel',
    #                 'note_vel_band', 'note_rel_vel', 'articulation', 'timing_dev', 
    #                 'keys', 'harmonic_quality']
    
    tokens_meta = ['beat', 'tempo_band', 'pitch', 'start', 'dur_full', 'ibi', 
                   'local_vel_mean', 'artic', 'timing_dev', 'note_vel_diff']
    
    if metric == 'histogram':
        for t_type in tokens_meta:
            unique_vals = get_unique_vals(tokens_path, t_type=t_type)
            histogram(unique_vals, save_path, t_type)

    if metric == 'print_unique_tokens':
        for t_type in tokens_meta:
            get_unique_vals(tokens_path, t_type=t_type, print_freqs=True)   

    if metric == 'print_unique_tokens_by_track':
        for t_type in tokens_meta:
            get_unique_vals(tokens_path, t_type=t_type, print_freqs=True, by_track=True) 
    
    if metric == 'vocab_lens':
        vocab_lens(tokens_path, tokens_meta)

    if metric == 'words_lens':
        named_len, lens, max_len = words_lens(words_meta_path)
        length_buckets = {1000 * i: 0 for i in range(21)}
        for l in lens:
            if l > 20500:
                print(f"Leangth out of range: {l}")
            else:
                length_buckets[int(utils.quantize(l, 1000))] += 1
        
        print(f"Maximum words sequence length: {max_len}")
        print("Lengths distribution: ")
        for k, v in length_buckets.items():
            print(f"\t{k}:\t{v}")

