"""
Token-related metrics
"""
import os
import pickle
import collections
import matplotlib.pyplot as plt
import sys
import json


def get_val_frequencies(tokens_root, t_type, verbose=True):
    """
    returns:
        unique_vals: dictionary(set) of {'token_type': {unique values across all tokens}}
    """
    unique_vals = dict()
    for entry in os.scandir(tokens_root):
        if verbose:
            print(f"Listing unique tokens for {entry} ...                     ", end='\r')
        with open(entry, 'rb') as f:
            token_dict = pickle.load(f)
        if type(t_type) is list:
            for token in token_dict[t_type].values():
                if token[1] in unique_vals:
                    unique_vals[token[1]] += 1
                else:
                    unique_vals[token[1]] = 0
        else:
            for tokens in token_dict[t_type].values():
                for token in tokens:
                    if token[1] in unique_vals:
                        unique_vals[token[1]] += 1
                    else:
                        unique_vals[token[1]] = 0
    
    return unique_vals


def histogram(data, save_path, name=None):
    """
    """
    data = sorted(data)
    plt.figure(dpi=100)
    plt.bar(list(data.keys()), data.values())
    if name:
        plt.title(f"Token frequencies for {name}")
        plt.xlabel(f"name")
    plt.ylabel('frequency')

    # save bar chart
    plt.savefig(save_path)
    plt.close()



if __name__ == '__main__':
    save_path = sys.argv[1]
    tokens_path = sys.argv[2]
    
    meta_path = os.path.join(tokens_path, '/metadata')
    with open(meta_path) as f:
        tokens_meta = json.load(f)
    
    for t_type in tokens_meta:
        unique_vals = get_val_frequencies(tokens_path, t_type=t_type)
        histogram(unique_vals, save_path)

