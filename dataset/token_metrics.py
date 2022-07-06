"""
Token-related metrics
"""
import os
import pickle
import collections


def get_val_frequencies(tokens_root, t_types, verbose=True):
    """
    returns:
        unique_vals: dictionary(set) of {'token_type': {unique values across all tokens}}
    """
    unique_vals = collections.defaultdict(dict)
    for entry in os.scandir(tokens_root):
        if verbose:
            print(f"Listing unique tokens for {entry} ...                     ", end='\r')
        with open(entry, 'rb') as f:
            token_dict = pickle.load(f)
        for t_type in t_types:
            for tokens in token_dict[t_type].values():
                for token in tokens:
                    if token[1] in unique_vals[t_type]:
                        unique_vals[t_type][token[1]] += 1
                    else:
                        unique_vals[t_type][token[1]] = 0


def histogram():
    pass

