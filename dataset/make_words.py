"""
References:
    https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/representations/uncond/cp/events2words.py
"""
import os
import pickle
import collections
import inspect
import sys

# import torch

import token_funcs


def word_pos(meta_len, metric_t_types, note_t_types):
    """
    Make dictionary of indices of word types within a word
    """
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
    
    return t_pos


def get_word_seq(tokens, cw, t_pos, 
                 eos_tokens, bar_tokens, type_tokens, 
                 metric_t_types, note_t_types):
    """
    create sequence of words for a single track
    """
    # initialise output
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
    
    return words


def compute_words(tokens_root, 
                  words_root, 
                  bar_tokens=None,
                  eos_tokens=(True, True),
                  type_tokens=True,
                  in_metric_t_types=[],
                  in_note_t_types=[],
                  attr_metric_t_types=[],
                  attr_note_t_types=[],
                  out_metric_t_types=[],
                  out_note_t_types=[]):
    """
    Convert tokens to compound words
    """
    os.makedirs(words_root, exist_ok=True)
    # meta_root = os.path.join(words_root, '/metadata')
    # os.makedirs(meta_root, exist_ok=True)
    
    # for looping
    t_types = in_metric_t_types + in_note_t_types + attr_metric_t_types + \
              attr_note_t_types + out_metric_t_types + out_note_t_types

    # loop through examples to get all unique token values for each token type    
    unique_vals = collections.defaultdict(set)
    for entry in os.scandir(tokens_root):
        print(f"Listing unique tokens for {entry} ...                     ", end='\r')
        with open(entry, 'rb') as f:
            token_dict = pickle.load(f)
        for t_type in in_metric_t_types + attr_metric_t_types + out_metric_t_types:
            for token in token_dict[t_type]:
                unique_vals[t_type].add(token[1])
        for t_type in in_note_t_types + attr_note_t_types + out_note_t_types:
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
                  [inspect.getfullargspec(compute_words)[0][-9:]]}
    # token_info = {
    #     'bar_tokens': bar_tokens,
    #     'eos_tokens': eos_tokens,
    #     'type_tokens': type_tokens,
    #     'metric_t_types': metric_t_types,
    #     'note_t_types': note_t_types
    # }
    
    # blank word templates
    meta_len = int(any(*eos_tokens, type_tokens))
    in_word_len = meta_len + len(in_metric_t_types) + len(in_note_t_types)
    attr_word_len = meta_len + len(attr_metric_t_types) + len(attr_note_t_types)
    out_word_len = meta_len + len(out_metric_t_types) + len(out_note_t_types)
    in_cw = [0] * in_word_len
    attr_cw = [0] * attr_word_len
    out_cw = [0] * out_word_len
    
    
    # token position in word
    in_pos = word_pos(meta_len, in_metric_t_types, in_note_t_types)
    attr_pos = word_pos(meta_len, attr_metric_t_types, attr_note_t_types)
    out_pos = word_pos(meta_len, out_metric_t_types, out_note_t_types)
    
    # for saving word lists
    all_in_words = []
    all_attr_words = []
    all_out_words = []
    
    # for saving word metadata
    words_info = {'names': [], 'lengths': []}
    
    # --- compile --- #
    # loop through examples to make word vectors of indices
    for entry in os.scandir(tokens_root):
        with open(entry, 'rb') as f:
            tokens = pickle.load(f)
        
        # words seq for track
        in_words = get_word_seq(tokens, in_cw, in_pos, 
                                eos_tokens, bar_tokens, type_tokens, 
                                in_metric_t_types, in_note_t_types)
        attr_words = get_word_seq(tokens, attr_cw, attr_pos, 
                                  eos_tokens, bar_tokens, type_tokens, 
                                  attr_metric_t_types, attr_note_t_types)
        out_words = get_word_seq(tokens, out_cw, out_pos, 
                                 eos_tokens, bar_tokens, type_tokens, 
                                 out_metric_t_types, out_note_t_types)
        
        assert len(in_words) == len(attr_words)
        assert len(attr_words) == len(out_words)
        
        # track name
        _, tail = os.path.split(entry)
        name = os.path.splitext(tail)[0]
        
        # update metadata
        # words_info[name] = dict()
        # w_len = len(in_words)
        # words_info[name]['num_words'] = w_len
        words_info['names'].append(name)
        words_info['lengths'].append(len(in_words))
        
        # add to lists
        all_in_words.append(in_words)
        all_attr_words.append(attr_words)
        all_out_words.append(out_words)
        
        # save words as tensor
        # words = torch.stack([torch.as_tensor(w) for w in words])
        # words = torch.as_tensor(words)
        # save_name = os.path.join(words_root, name + '.pt')
        # torch.save(words, save_name)
    
    # save metadata
    metadata = {}
    metadata['token_info'] = token_info
    metadata['words_info'] = words_info
    metadata['in_token_positions'] = in_pos
    metadata['attr_token_positions'] = attr_pos
    metadata['out_token_positions'] = out_pos
    
    meta_name = os.path.join(words_root, 'metadata.pkl')
    with open(meta_name, 'wb') as f:
        pickle.dump((val2idx, idx2val, metadata), f)
    
    # save data
    words_name = os.path.join(words_root, 'words.pkl')
    with open(words_name, 'wb') as f:
        pickle.dump((all_in_words, all_attr_words, all_out_words), f)
    

if __name__ == '__main__':
    tokens_root = sys.argv[1]
    words_root = sys.argv[2]
    
    compute_words(tokens_root, 
                  words_root, 
                  bar_tokens=None,
                  eos_tokens=(True, True),
                  type_tokens=True,
                  in_metric_t_types=[],
                  in_note_t_types=[],
                  attr_metric_t_types=[],
                  attr_note_t_types=[],
                  out_metric_t_types=[],
                  out_note_t_types=[])
    
    