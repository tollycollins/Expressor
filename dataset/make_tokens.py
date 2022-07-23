"""
Create dictionary of tokens

References:
    https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/representations/uncond/cp/corpus2events.py
"""
import os
import pickle
import itertools
import utils
import json

import pretty_midi

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


def remove_invalid_beats(metadata, eps=1e-3):
    """
    Remove items of metadata where either performance or score beat annotations 
        have two different beats annotated at the same time point, or where the
        IBI < eps
        
    Returns:
        valid, invalid
    """
    valid = dict()
    invalid = dict()
    for k, v in metadata.items():
        valid[k] = v
        ibis = utils.ibis(v['performance_beats'])
        ibis.extend(utils.ibis(v['midi_score_beats']))
        for ibi in ibis:
            if ibi < eps:
                invalid.update(valid.popitem())
                break

    return valid, invalid


def compute_tokens(t_types: dict, 
                   annot,
                   tokens_path, 
                   score_path, 
                   corpus_sigs=None,
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
    score = pretty_midi.PrettyMIDI(os.path.join('dataset/asap-dataset', score_path))
    perf = pretty_midi.PrettyMIDI(os.path.join('dataset/asap-dataset', perf_path))
    
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
                         'note_vel_band', 'note_rel_vel',
                         'note_vel_diff'] if t in t_types]
    if len(dynam):   
        kwargs = dict(list(itertools.chain(*[list(t_types[i].items()) for i in dynam])))
        local_vel_mean, local_vel_std, \
            local_vel_band, note_rel_vel, \
            note_vel, note_vel_band, \
            note_vel_diff = token_funcs.dynamics_tokens(perf_gp,
                                                        beats_score,
                                                        **kwargs)
        for t in dynam:
            tokens[t] = locals()[t]
    
    # expressive timing labels
    timing = [t for t in ['artic', 'artic_whole', 'artic_fract', 'timing_dev'
                          'timing_dev_whole', 'timing_dev_fract'] if t in t_types]
    if len(timing):
        if 'duration' in t_types:
            duration = t_types['duration']
        elif duration is None:
            _, _, duration, _, _ = token_funcs.note_tokens(score_notes, beats_score)
        kwargs = dict(list(itertools.chain(*[list(t_types[i].items()) for i in timing])))
        artic, artic_whole, artic_fract, \
            timing_dev, timing_dev_whole, \
            timing_dev_fract = token_funcs.timing_labels(score_gp,
                                                         perf_gp,
                                                         duration,
                                                         beats_score,
                                                         beats_perf,
                                                         **kwargs)
        for t in timing:
            tokens[t] = locals()[t]
    
    # harmonic tokens
    harm = [t for t in ['keys', 'harmonic_quality'] if t in t_types]
    if len(harm):
        # if 'time_sig' in t_types:
        #     time_sig = t_types['time_sig']
        # elif time_sig is None:
        #     time_sig = token_funcs.time_sig_tokens(beats_score, utils.get_time_sigs(score),
        #                                            corpus_sigs)
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


def create_training_data(t_types, tokens_base, align_range=0.25, beat_eps=1e-2):
    """
    Save data and corresponding metadata file
    """
    os.makedirs(tokens_base, exist_ok=True)
    meta_path = os.path.join(tokens_base, 'metadata')
    os.makedirs(meta_path, exist_ok=True)

    metadata = utils.get_metadata()
    
    # only keep items in both score and performance
    metadata, invalid = remove_non_aligned(metadata)
    print(f"{len(invalid)} non-aligned performances removed from dataset")

    # only keep items with valid beat annotations
    metadata, invalid = remove_invalid_beats(metadata, beat_eps)
    print(f"{len(invalid)} tracks with invalid beat annotations removed from dataset")
    
    # get list of all time signatures in corpus
    corpus_sigs = None
    if 'time_sig' in t_types:
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
        
        compute_tokens(t_types, annot, tokens_path, score_path, corpus_sigs=corpus_sigs,
                       perf_path=perf_path, align_range=align_range)
    
    print("\nProcessing complete\n")
    
    path = os.path.join(meta_path, 'tokens_meta.json')
    try:
        with open(path) as f:
            tokens_meta = json.load(f)
    except FileNotFoundError:
        tokens_meta = {}
    
    tokens_meta.update(t_types)
    tokens_meta['align_range'] = align_range
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(tokens_meta, f, ensure_ascii=False)


if __name__ == '__main__':
    """
    Available token types and kwargs [copy and paste into dictionary in function below]:
        'bar': {},
        'beat': {},
        'ibi': {'ibi_tokens': True},
        'tempo_band': {'lower_bounds': default_tempo_lower_bounds, 'hysteresis': default_tempo_hysteresis, 'allow_zero': default_tempo_allow_zero},
        'local_tempo': {'lower_bounds': default_tempo_lower_bounds, 'hysteresis': default_tempo_hysteresis, 'allow_zero': default_tempo_allow_zero, 
                        'median_time': 4, 'local_tempo_quant': 1},
        'time_sig': {'allowed_time_sigs': None, 'allow_other': True, 'min_freq': 0, 'conti': False},
        'pitch': {},
        'start': {'start_quant': 1/60},
        'duration': {'duration_quant': default_duration_quant},
        'dur_full': {'duration_quant': default_duration_quant, 'split_duration': True},
        'dur_fract': {'duration_quant': default_duration_quant, 'split_duration': True},
        'local_vel_band': {'mean_len': default_dynamics_mean_len, 'bands': default_dynamics_bands, 'band_win': 1, 'band_hysteresis': (5, 5)},
        'local_vel_mean': {'mean_len': default_dynamics_mean_len, 'mean_quant': 1},
        'local_vel_std': {'mean_len': default_dynamics_mean_len, 'std_quant': 1},
        'note_vel': {},
        'note_vel_band': {'bands': default_dynamics_bands, 'band_hysteresis': (5, 5)},
        'note_rel_vel': {'note_std_quant': 0.2, 'note_std_bounds': (-15, 15)},
        'artic': {'artic_quant': default_artic_quant, 'artic_lims': default_artic_lims, 'calc_type': default_timimng_calc_type},
        'artic_whole': {'artic_quant': default_artic_quant, 'artic_lims': default_artic_lims, 'calc_type': default_timimng_calc_type},
        'artic_fract': {'artic_quant': default_artic_quant, 'artic_lims': default_artic_lims, 'calc_type': default_timimng_calc_type},
        'timing_dev': {'dev_quant': default_dev_quant, 'dev_lims': default_dev_lims, 'cubic_len': default_cubic_len, 
                       'beat_in_beat_weight': default_beat_in_beat_weight, 'non_beat_in_beat_weight': default_non_beat_in_beat_weight,
                       'calc_type': default_timing_calc_type},
        'timing_dev_whole': {'dev_quant': default_dev_quant, 'dev_lims': default_dev_lims, 'cubic_len': default_cubic_len, 
                             'beat_in_beat_weight': default_beat_in_beat_weight, 'non_beat_in_beat_weight': default_non_beat_in_beat_weight,
                             'calc_type': default_timing_calc_type},
        'timing_dev_fract': {'dev_quant': default_dev_quant, 'dev_lims': default_dev_lims, 'cubic_len': default_cubic_len, 
                             'beat_in_beat_weight': default_beat_in_beat_weight, 'non_beat_in_beat_weight': default_non_beat_in_beat_weight,
                             'calc_type': default_timing_calc_type},
        'keys': {'reduce': default_harmonic_reduce, 'extra_reduce': default_harmonic_extra_reduce, 'conti': default_harmonic_conti},
        'harmonic_quality': {'reduce': default_harmonic_reduce, 'extra_reduce': default_harmonic_extra_reduce, 'conti': default_harmonic_conti},
        'rubato': {}
    """
    default_tempo_lower_bounds = [5, 45, 85, 120, 150]
    default_tempo_hysteresis = [5, 5]
    default_tempo_allow_zero = False

    default_duration_quant = 1/60

    default_dynamics_mean_len = 1
    default_dynamics_bands = [0, 31, 63, 95, 127]
    default_dynamics_band_hysteresis = (5, 5)

    default_artic_quant = 0.05
    default_artic_lims = None
    default_timing_calc_type = 'linear'
    default_dev_quant = 0.01
    default_dev_lims = None
    default_cubic_len = 6
    default_beat_in_beat_weight = 1
    default_non_beat_in_beat_weight = 2
    
    default_harmonic_reduce = False
    default_harmonic_extra_reduce = False
    default_harmonic_conti = True
    
    create_training_data({'bar': {},
                        'beat': {},
                        'ibi': {'ibi_tokens': True},
                        'tempo_band': {'lower_bounds': default_tempo_lower_bounds, 'hysteresis': default_tempo_hysteresis, 'allow_zero': default_tempo_allow_zero},
                        'local_tempo': {'lower_bounds': default_tempo_lower_bounds, 'hysteresis': default_tempo_hysteresis, 'allow_zero': default_tempo_allow_zero, 
                                        'median_time': 4, 'local_tempo_quant': 1},
                        'time_sig': {'allowed_time_sigs': None, 'allow_other': True, 'min_freq': 0, 'conti': False},
                        'pitch': {},
                        'start': {'start_quant': 1/60},
                        'duration': {'duration_quant': default_duration_quant},
                        'dur_full': {'duration_quant': default_duration_quant, 'split_duration': True},
                        'dur_fract': {'duration_quant': default_duration_quant, 'split_duration': True},
                        'local_vel_band': {'mean_len': default_dynamics_mean_len, 'bands': default_dynamics_bands, 'band_win': 1, 'band_hysteresis': (5, 5)},
                        'local_vel_mean': {'mean_len': default_dynamics_mean_len, 'mean_quant': 1},
                        'local_vel_std': {'mean_len': default_dynamics_mean_len, 'std_quant': 1},
                        'note_vel': {},
                        'note_vel_band': {'bands': default_dynamics_bands, 'band_hysteresis': (5, 5)},
                        'note_rel_vel': {'note_std_quant': 0.2, 'note_std_bounds': (-15, 15)},
                        'artic': {'artic_quant': default_artic_quant, 'artic_lims': default_artic_lims, 'calc_type': default_timing_calc_type},
                        'artic_whole': {'artic_quant': default_artic_quant, 'artic_lims': default_artic_lims, 'calc_type': default_timing_calc_type},
                        'artic_fract': {'artic_quant': default_artic_quant, 'artic_lims': default_artic_lims, 'calc_type': default_timing_calc_type},
                        'timing_dev': {'dev_quant': default_dev_quant, 'dev_lims': default_dev_lims, 'cubic_len': default_cubic_len, 
                                       'beat_in_beat_weight': default_beat_in_beat_weight, 'non_beat_in_beat_weight': default_non_beat_in_beat_weight,
                                       'calc_type': default_timing_calc_type},
                        'timing_dev_whole': {'dev_quant': default_dev_quant, 'dev_lims': default_dev_lims, 'cubic_len': default_cubic_len, 
                                             'beat_in_beat_weight': default_beat_in_beat_weight, 'non_beat_in_beat_weight': default_non_beat_in_beat_weight,
                                             'calc_type': default_timing_calc_type},
                        'timing_dev_fract': {'dev_quant': default_dev_quant, 'dev_lims': default_dev_lims, 'cubic_len': default_cubic_len, 
                                             'beat_in_beat_weight': default_beat_in_beat_weight, 'non_beat_in_beat_weight': default_non_beat_in_beat_weight,
                                             'calc_type': default_timing_calc_type},
                        'keys': {'reduce': default_harmonic_reduce, 'extra_reduce': default_harmonic_extra_reduce, 'conti': default_harmonic_conti},
                        'harmonic_quality': {'reduce': default_harmonic_reduce, 'extra_reduce': default_harmonic_extra_reduce, 'conti': default_harmonic_conti}}, 
                         'dataset/tokens/t1', align_range=0.25, beat_eps=1e-2)

