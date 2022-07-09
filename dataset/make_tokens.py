"""
Create dictionary of tokens

References:
    https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/representations/uncond/cp/corpus2events.py
"""
import os
import pickle
import itertools
import utils
import sys
import json

import pretty_midi
import miditoolkit

import token_funcs
import asap_metrics


def remove_non_aligned(metadata):
    """
    Remove items of metadata without matching score and performance beat annotations

    Returns:
        aligned: All items with matching annotations
        non_aligned: All items without metching annotations
    """
    aligned = {k: v for k, v in metadata.items() if v['score_and_performance_aligned']}
    non_aligned = {k: v for k, v in metadata.items()
                   if not v['score_and_performance_aligned']}
    return aligned, non_aligned



def compute_tokens(t_types: dict, 
                   annot,
                   corpus_sigs,
                   tokens_path, 
                   score_path, 
                   perf_path=None,
                   align_range=0.25):
    """
    Convert MIDI file of human performance to individual REMI-like tokens, expressive 
        attributes and expressive ground truth labels.  
    For training purposes.  

    Select which tokens to add to the data dictionary [or to update]

    Create metadata
    
    args:
        t_types: {'name': ([kwargs]} - dict of token names to be added / updated
                                       and optional kwargs for relevant function
        annot: annotations entry from metadata for this performance
        tokens_path: path to current saved dict
        score_path: path to score MIDI file
        perf_path: path to performance MIDI file
    """
    # load MIDI
    score = pretty_midi.PrettyMIDI(os.path.join('asap-dataset', score_path))
    perf = pretty_midi.PrettyMIDI(os.path.join('asap-dataset', perf_path))
    
    # mtk_score = miditoolkit.midi.parser.MidiFile(os.path.join('asap-dataset', score_path))
    
    # load current data
    try:
        with open(tokens_path, 'rb') as f:
            tokens = pickle.load(f)
    except FileNotFoundError:
        tokens = dict()
    
    # get beat annotations
    db_score = annot['midi_score_downbeats']
    beats_score = annot['midi_score_beats']
    beats_perf = annot['performance_beats']
    
    # get note objects
    score_notes = list(itertools.chain(*[i.notes for i in score.instruments]))
    perf_notes = list(itertools.chain(*[i.notes for i in perf.instruments]))

    # align score and performance notes
    score_notes, perf_notes, _, _ = utils.align_notes(score_notes,
                                                      perf_notes,
                                                      beats_score,
                                                      beats_perf,
                                                      search_range=align_range)
    
    # goup notes by score beat
    score_gp, perf_gp = utils.group_by_beat(beats_score,
                                            notes=score_notes,
                                            aligned_notes=perf_notes)
    
    # --- get tokens --- #
    # metre tokens
    if 'bar' in t_types:
        tokens['bar'] = token_funcs.bar_tokens(beats_score, db_score)
    
    if 'beat' in t_types:
        tokens['beat'] = token_funcs.beat_tokens(beats_score, db_score)
    
    # tempo tokens
    tempi = [t for t in ['ibi', 'tempo_band', 'local_tempo'] if t in t_types]
    if len(tempi):
        kwargs = dict(list(itertools.chain(*[list(t_types[i].items()) for i in tempi])))
        ibi, tempo_band, local_tempo = \
            token_funcs.tempo_tokens(beats_score,
                                     ibi_tokens=('ibi' in tempi),
                                     band_tokens=('tempo_band' in tempi),
                                     local_tempo_tokens=('local_tempo' in tempi),
                                     **kwargs)
        for t in tempi:
            tokens[t] = locals()[t]
    
    # time signature tokens
    time_sig = None
    if 'time_sig' in t_types:
        tokens['time_sig'] = token_funcs.time_sig_tokens(beats_score, 
                                                         utils.get_time_sigs(score),
                                                         corpus_sigs,
                                                         **t_types['time_sig'])
    
    # note tokens
    note_tok = [t for t in ['pitch', 'start', 'duration', 
                            'dur_full', 'dur_fract'] if t in t_types]
    duration = None
    if len(note_tok):
        kwargs = dict(list(itertools.chain(*[list(t_types[i].items()) for i in note_tok])))
        pitch, start, duration, dur_full, dur_fract = token_funcs.note_tokens(score_notes,
                                                                              beats_score,
                                                                              **kwargs)
        for t in note_tok:
            tokens[t] = locals()[t]
    
    # dynamics tokens
    dynam = [t for t in ['local_vel_band', 'local_vel_mean', 
                         'local_vel_std', 'note_vel',
                         'note_vel_band', 'note_rel_vel'] if t in t_types]
    if len(dynam):   
        kwargs = dict(list(itertools.chain(*[list(t_types[i].items()) for i in dynam])))
        local_vel_mean, local_vel_std, \
            local_vel_band, note_rel_vel, \
            note_vel, note_vel_band = token_funcs.dynamics_tokens(perf_gp,
                                                                      beats_score,
                                                                      **kwargs)
        for t in dynam:
            tokens[t] = locals()[t]
    
    # expressive timing labels
    timing = [t for t in ['articulation', 'timing_dev'] if t in t_types]
    if len(timing):
        if 'duration' in t_types:
            duration = t_types['duration']
        elif duration is None:
            _, _, duration, _, _ = token_funcs.note_tokens(score_notes, beats_score)
        kwargs = dict(list(itertools.chain(*[list(t_types[i].items()) for i in timing])))
        articulation, timing_dev = token_funcs.timing_labels(score_notes,
                                                             perf_notes,
                                                             duration,
                                                             beats_score,
                                                             beats_perf,
                                                             **kwargs)
        for t in timing:
            tokens[t] = locals()[t]
    
    # harmonic tokens
    harm = [t for t in ['keys', 'harmonic_quality'] if t in t_types]
    if len(harm):
        if 'time_sig' in t_types:
            time_sig = t_types['duration']
        elif time_sig is None:
            time_sig = token_funcs.time_sig_tokens(beats_score, utils.get_time_sigs(score),
                                                   corpus_sigs,)
        kwargs = dict(list(itertools.chain(*[list(t_types[i].items()) for i in harm])))
        keys, harmonic_quality = token_funcs.harmonic_tokens(score_gp,
                                                             beats_score,
                                                             **kwargs)
        for t in harm:
            tokens[t] = locals()[t]
    
    # --- latent space attribute tokens --- #
    # rubato tokens
    if 'rubato' in t_types:
        tokens['rubato'] = token_funcs.rubato_tokens(beats_score, 
                                                     annot['performance_beats_type'])

    # metatdata (save t_types)
    if 'metadata' not in tokens:
        tokens['metadata'] = dict()
    tokens['metadata'].update(t_types)
    
    # save data
    with open(tokens_path, 'wb') as f:
        pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)


def create_training_data(t_types, tokens_base, align_range=0.25):
    """
    Save data and corresponding metadata file
    """
    os.makedirs(tokens_base, exist_ok=True)
    meta_path = os.path.join(tokens_base, 'metadata')
    os.makedirs(meta_path, exist_ok=True)

    metadata = utils.get_metadata()
    
    # only keep items in both score and performance
    metadata, _ = remove_non_aligned(metadata)

    # get list of all time signatures in corpus
    corpus_sigs = asap_metrics.time_signature(verbose=False)

    print(f"Updating {list(t_types.keys())} at: {tokens_base}\n")
    
    names = []
    for idx, (perf_path, annot) in enumerate(metadata.items()):
        # paths
        head, tail = os.path.split(perf_path)
        name = os.path.splitext(tail)[0]
        
        # deal with duplicate names
        name += '_0'
        while name in names:
            name = name[:-1] + str(int(name[-1]) + 1)
        names.append(name)
        
        token_name = name + '.pkl'
        tokens_path = os.path.join(tokens_base, token_name)

        score_path = os.path.join(head, 'midi_score.mid')

        print(f"Processing {idx + 1} out of {len(metadata)}: {name} ...        ", end='\r')
        
        compute_tokens(t_types, annot, corpus_sigs, tokens_path, score_path, 
                       perf_path=perf_path, align_range=align_range)
    
    print("\nProcessing complete\n")
    
    path = os.path.join(meta_path, 'tokens_meta.json')
    try:
        with open(path) as f:
            tokens_meta = json.load(f)
    except FileNotFoundError:
        tokens_meta = dict()
    
    tokens_meta.update(t_types)
    
    with open(path, 'w') as f:
        json.dump(tokens_meta, f)


if __name__ == '__main__':
    # create_training_data({'local_vel_band': {}, 'local_vel_mean': {}, 
    #                      'local_vel_std': {}, 'note_vel': {},
    #                      'note_vel_band': {}, 'note_rel_vel': {}, 'time_sig': {}}, 'tokens/t1')

    create_training_data({'local_vel_mean': {}, 
                          'local_vel_std': {}, 'note_vel': {},
                          'note_vel_band': {}, 'note_rel_vel': {}, 'time_sig': {}}, 
                          'tokens/t1')

