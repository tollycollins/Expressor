"""
References:
    https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/representations/uncond/cp/events2words.py
"""
import os
import pickle
import collections
import inspect
import sys

import torch

import token_funcs


def compute_words(tokens_root, 
                  words_root, 
                  bar_tokens=None,
                  eos_tokens=(True, True),
                  type_tokens=True,
                  metric_t_types=[],
                  note_t_types=[]):
    """
    Convert tokens to compound words
    """
    os.makedirs(words_root, exist_ok=True)
    meta_root = os.path.join(words_root, '/metadata')
    os.makedirs(meta_root, exist_ok=True)
    
    # for looping
    t_types = metric_t_types.extend(note_t_types)

    # loop through examples to get all unique token values for each token type    
    unique_vals = collections.defaultdict(set)
    for entry in os.scandir(tokens_root):
        print(f"Listing unique tokens for {entry} ...                     ", end='\r')
        with open(entry, 'rb') as f:
            token_dict = pickle.load(f)
        for t_type in metric_t_types:
            for token in token_dict[t_type]:
                unique_vals[t_type].add(token[1])
        for t_type in note_t_types:
            for tokens in token_dict[t_type].values():
                for token in tokens:
                    unique_vals[t_type].add(token[1])
    if bar_tokens is not None:
        # give value of '-1' for a bar token (treated as a special beat token)
        unique_vals['beat'].add(-1)
    
    # dictionaries for converting between values and token indices
    val2idx = dict()
    idx2val = dict()
    for t_type in t_types:
        vals = sorted(unique_vals[t_type])
        val2idx[t_type] = {val: idx for idx, val in enumerate(vals)}
        idx2val[t_type] = {idx: val for idx, val in enumerate(vals)}
    
    # print
    print("\nVocab sizes: \n")
    for t_type in t_types:
        print(f"{t_type}: {len(unique_vals[t_type])}")
    
    # token metadata
    token_info = {arg: inspect.getargvalues(inspect.currentframe())[3][arg] for arg in 
                  [inspect.getfullargspec(compute_words)[0][-5:]]}
    # token_info = {
    #     'bar_tokens': bar_tokens,
    #     'eos_tokens': eos_tokens,
    #     'type_tokens': type_tokens,
    #     'metric_t_types': metric_t_types,
    #     'note_t_types': note_t_types
    # }
    
    # word template
    meta_len = int(any(*eos_tokens, type_tokens))
    word_len = meta_len + len(metric_t_types) + len(note_t_types)
    cw = [0] * word_len

    # token position in word
    ptr = 0
    t_pos = {}
    if meta_len:
        t_pos['meta'] = 0
        ptr += 1
    for t_type in metric_t_types:
        t_pos[t_type] = ptr
        ptr += 1
    for t_type in note_t_types:
        t_pos[t_type] = ptr
        ptr += 1
    
    # for saving word metadata
    words_info = {}

    # --- compile --- #
    # loop through examples to make word vectors of indices
    for entry in os.scandir(tokens_root):
        with open(entry, 'rb') as f:
            tokens = pickle.load(f)
        
        # all words for track
        words = []

        # beginning of sequence token
        if eos_tokens[0]:
            word = cw.copy()
            word[0] = token_funcs.type_token_val('eos')
            words.append(word)
        
        # words by beat
        for idx in range(len(tokens['beat'] + 1)):
            # --- metric word -- #
            # don't consider anacrisus for beat tokens
            if idx > 0:
                word = cw.copy()

                # bar token
                if bar_tokens is not None and tokens['bar'][idx - 1][1] == 1:
                    if type_tokens:
                        word[0] = token_funcs.type_token_val('metric')
                    # special beat value for bars
                    word[t_pos['beat']] = -1
                    # add word to list
                    words.append(word)
                    # reset word
                    word = cw.copy()
                
                # type metatoken
                if type_tokens:
                    word[0] = token_funcs.type_token_val('metric')
                
                # metric tokens
                for t_type in metric_t_types:
                    word[t_pos[t_type]] = tokens[t_type][idx - 1]
                
                # add word
                words.append(word)

            # --- note words --- #
            # loop over notes in beat
            for note in range(len(tokens[note_t_types[0]][idx])):
                word = cw.copy()
                
                # type metatoken
                if type_tokens:
                    word[0] = token_funcs.type_token_val('note')
                
                # note nokens
                for t_type in note_t_types:
                    word[t_pos[t_type]] = tokens[t_type][idx][note]

                # add word
                words.append(word)
            
        # eos token
        if eos_tokens[1]:
            word = cw.copy()
            word[0] = token_funcs.type_token_val('eos')
            words.append(word)
        
        # track name
        _, tail = os.path.split(entry)
        name = os.path.splitext(tail)[0]
        
        # update metadata
        words_info[name] = dict()
        w_len = len(words)
        words_info[name]['num_words'] = w_len
        
        # save words list
        words = torch.as_tensor(words)
        save_name = os.path.join(words_root, name + '.pt')
        torch.save(words, save_name)
    
    # save metadata
    metadata = {}
    metadata['token_info'] = token_info
    metadata['words_info'] = words_info
    metadata['token_positions'] = t_pos
    
    path = os.path.join(meta_root, 'metadata.pkl')
    with open(path, 'wb') as f:
        pickle.dump((val2idx, idx2val, metadata), f)

    

if __name__ == '__main__':
    tokens_root = sys.argv[1]
    words_root = sys.argv[2]
    
    compute_words(tokens_root, 
                  words_root, 
                  bar_tokens=None,
                  eos_tokens=(True, True),
                  type_tokens=True,
                  metric_t_types=[],
                  note_t_types=[])
    
    