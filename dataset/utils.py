"""
Utility function for token creation.  
"""


from multiprocessing.dummy import Value
import os
from pathlib import Path
import json
import re
from decimal import Decimal
import copy
import itertools

import numpy as np
from scipy.interpolate import interp1d


def get_metadata():
    """
    Return ASAP metadata file as a dictionary
    Keys: path for MIDI performance
    Values are dictionaries with the following keys:
        performance_beats
        performance_downbeats
        performance_beats_type
        perf_time_signatures
        perf_key_signatures
        midi_score_beats
        midi_score_downbeats
        midi_score_beats_type
        midi_score_time_signatures
        midi_score_key_signatures
        downbeats_score_map
        score_and_performance_aligned
    """
    meta_path = 'asap-dataset/asap_annotations.json'
    json_file = None
    with open(os.path.join(Path(__file__).resolve().parent, meta_path)) as file:
        json_file = json.load(file)
    return json_file


def get_time_sigs(pm_midi):
    """
    return time signatures list in correct format for tokenization
        from a pretty_midi object
    """
    ts_changes = pm_midi.time_signature_changes
    time_sigs = []
    for ts in ts_changes:
        time_sig = str(ts.numerator) + '/' + str(ts.denominator)
        time_sigs.append((ts.time, time_sig))
            
    return time_sigs


def quantize(value, factor):
    """
    Round values to a given quantization interval
    """
    # get number of dicimal places for factor
    dp = str(factor)[::-1].find('.')
    # if there was no decimal point
    if dp == -1:
        dp = 0
    return round(round(value / factor) * factor, dp)


def bound(value, bounds):
    """
    Limit a value to fit in between 2 bounds
    args:
        value: float
        bounds (min, max): hard limits
    """
    value = bounds[0] if value < bounds[0] else value
    value = bounds[1] if value > bounds[1] else value
    return value


class Band():
    """
    Place value in a band, with optional hysteresis
    """
    def __init__(self,
                 initial_value=None,
                 bounds=None,
                 upper_bound=False,
                 hysteresis=(0, 0)):
        """
        initial_value: initialise with a value to set current band
        bounds: band boundaries
        upper_bound: if False, all bounds are taken to be lower bounds, and 
                     any value above the highest bound value will be allocated
                     to the highest class.
                     if True, the final value of 'bounds' is taken to be the 
                     largest possible value
        hysteresis (lower, upper): thresholds for moving to the next band
        """
        self.bounds = bounds
        self.upper_bound = upper_bound
        self.hysteresis = hysteresis
        
        # initialise band
        self.band = None
        if initial_value is not None:
            self.new_band(initial_value)
    
    def new_band(self, value):
        # check value
        value = self.validate(value, test=False)
        # initialise to highest band
        num_bands = len(self.bounds) if not self.upper_bound else len(self.bounds) - 1
        self.band = num_bands - 1
        # check against band boundaries
        for i in range(1, num_bands):
            if value < self.bounds[i]:
                self.band = i - 1
                break
        return self.band

    def update(self, value):
        # check value
        value = self.validate(value, test=False)
        # update band
        # check if we have a previous value
        if self.band is None:
            self.new_band(value)
        # check if value is within hysteresis bounds
        elif value < self.bounds[self.band] - self.hysteresis[0]:
            self.new_band(value)
        elif not self.upper_bound and self.band == len(self.bounds) - 1:
            pass
        elif value > self.bounds[self.band + 1] + self.hysteresis[1]:
            self.new_band(value)
        
        return self.band

    def validate(self, value, **kwargs):
        if value < self.bounds[0] or (self.upper_bound and value > self.bounds[-1]):
            raise RuntimeError(f"Value {value} out of bounds")  
        return value  


def ibis(beats):
    """
    Calculate inter-beat intervals
    args:
        beats: list of beat times (s)
    """
    ibi = np.squeeze(np.diff(beats))
    return np.concatenate((ibi, [ibi[-1]])).tolist()


def time_to_bpm(interval):
    """
    Beat interval in secs to BPM converter
    """
    return 60 / interval


class TempoBand(Band):
    """
    Holds current tempo band as an integer
    Note: '0' refers to the slowest band
    """
    def __init__(self, 
                 tempo=6, 
                 lower_bounds=[5, 45, 85, 120, 150],
                 hysteresis=(5, 5),
                 input_secs=True,
                 allow_zero=False):
        """
        Args:
            tempo: starting tempo
            lower_bounds: lower bounds of each tempo band (BPM)
            hysteresis: hysteresis for band transition (BPM)
            input_secs: convert all inputs from secs to BPM
            allow_zero: a zero tempo will not throw an error (if working in BPM)

        note: values in secs will only be converted to tempi if they last for 
              less time than the minimum tempo band lower bound
        """
        self.input_secs = input_secs
        self.converted = False
        
        self.allow_zero = allow_zero
        
        super().__init__(initial_value=tempo,
                         bounds=lower_bounds,
                         hysteresis=hysteresis)
    
    def new_band(self, value):
        value = super().new_band(value)
        self.converted = False
        return value
    
    def update(self, value):
        value = super().update(value)
        self.converted = False
        return value
    
    def validate(self, tempo, test=True):
        """
        Use test=True to call validate directly
        """
        if tempo < 0:
            raise RuntimeError("Negative tempo provided")
        elif (not self.allow_zero or self.input_secs) and tempo == 0:
            raise RuntimeError("Tempo value of zero provided")
        # convert secs to BPM if necessary
        if self.input_secs and not self.converted and not test:
            tempo = time_to_bpm(tempo)
            self.converted = True
        return tempo


def group_by_beat(beats, 
                  tokens=None, 
                  notes=None, 
                  add_flag=False,
                  aligned_tokens=None,
                  aligned_notes=None):
    """
    args:
        beats: list of beat times
        tokens: list of tokens in intended order
        notes: list of note objects, can be out of order
        add_flag: add a False entry to the end of each token

    Note: all tokens are assumed to be tuples with 'time' as the first item
    
    returns:
        list of lists of tokens, grouped by beat
    """
    
    groups = [[] for i in range(len(beats) + 1)]
    aligned_groups = None
    if aligned_tokens or aligned_notes:
        aligned_groups = copy.deepcopy(groups)
    ptr = 0
    
    def add_token(gps, gp_num, token):
        token = tuple([*token, False]) if add_flag else token
        gps[gp_num].append(token)
    
    def add_note(gps, gp_num, note):
        if add_flag:
            note.flag = False
        gps[gp_num].append(note)
    
    if tokens is not None:
        # loop through beats
        for i, time in enumerate(beats):
            # while tokens remain
            while ptr < len(tokens) and tokens[ptr][0] < time:
                add_token(groups, i, tokens[ptr])
                # add aligned tokens
                if aligned_tokens is not None:
                    add_token(aligned_groups, i, aligned_tokens[ptr])
                ptr += 1
        
        # deal with any tokens after the final beat
        if ptr < len(tokens):
            for i in range(ptr, len(tokens)):
                add_token(groups, -1, tokens[i])
                if aligned_tokens is not None:
                    add_token(aligned_groups, -1, aligned_tokens[i])
    
    elif notes is not None:
        # sort notes
        notes = sorted(notes, key=lambda x: (x.start, x.pitch))
        
        # loop through beats
        for i, time in enumerate(beats):
            # while notes remain
            while ptr < len(notes) and notes[ptr].start < time:
                add_note(groups, i, notes[ptr])
                # add aligned tokens
                if aligned_notes is not None:
                    add_note(aligned_groups, i, aligned_notes[ptr])
                ptr += 1
            
        # deal with any notes after the final beat
        if ptr < len(notes):
            for i in range(ptr, len(notes)):
                add_note(groups, -1, notes[i])
                if aligned_notes is not None:
                    add_note(aligned_groups, -1, aligned_notes[i])
    
    else:
        raise RuntimeError("Either 'tokens' or 'notes' should be assigned a list")
    
    return groups, aligned_groups


def align_notes(notes_score, notes_perf, beats_score, beats_perf,
                search_range=0.25):
    """
    Note: beat annotations for score and performance must match
    Note: align in order relative to score

    args:
        search_range: proportion of a beat to search for match (+-)
    returns:
        aligned_score, aligned_perf: Aligned lists of note objects, ordered by 
            start time, then pitch (lowest to highest)
        score_extras: notes in the score, not present in the performance 
        perf_extras: notes in the performance, not present in the score
    """
    # group notes into beats
    gps_score, _ = group_by_beat(beats_score, notes=notes_score, add_flag=False)
    gps_perf, _ = group_by_beat(beats_perf, notes=notes_perf, add_flag=True)
    
    # add extra beats at start and end
    beats_score = beats_score = [0] + beats_score + \
                  [2 * beats_score[-1] - beats_score[-2]]
    beats_perf = beats_perf = [0] + beats_perf + \
                 [2 * beats_perf[-1] - beats_perf[-2]]
    
    # get IBIs
    ibis_score = ibis(beats_score)
    ibis_perf = ibis(beats_perf)
    # add IBI value for beat 0
    ibis_score = [ibis_score[0]] + ibis_score
    ibis_perf = [ibis_perf[0]] + ibis_perf
    
    # initialise outputs
    aligned_score, aligned_perf, score_extras, perf_extras = [], [], [], []
    
    # loop through beats in score
    for beat, group in enumerate(gps_score):
        # loop through notes in beat
        for note in group:
            # locate matching note in performance
            search_idx = slice(max(0, beat - 1), min(len(gps_perf), beat + 2))
            notes = list(itertools.chain(*gps_perf[search_idx]))
            # notes = [*gps_perf[search_idx]]

            # find expected time value in performed score
            # ratio = (note.start - beats_score[beat]) / ibis_score[beat] if \
            #         ibis_score[beat] not in [0, np.inf] else 0
            # exp_time = beats_perf[beat] + ratio * ibis_perf[beat] if \
            #            ibis_perf[beat] not in [0, np.inf] else beats_perf[beat]
            ratio = (note.start - beats_score[beat]) / ibis_score[beat]
            exp_time = beats_perf[beat] + ratio * ibis_perf[beat]


            # find min and max allowable time values
            bounds = (exp_time - ibis_perf[beat] * search_range, 
                      exp_time + ibis_perf[beat] * search_range)
                      
            # check notes
            found = False
            for n in notes:
                # check if it has already been aligned
                if n.flag or n.start < bounds[0]:
                    continue
                # check if we have run out of candidates
                if n.start > bounds[1]:
                    break
                # check pitch
                if n.pitch == note.pitch:
                    aligned_score.append(note)
                    aligned_perf.append(n)
                    n.flag = True
                    found = True
                    break
            
            if not found:
                score_extras.append(note)
        
    # check all performed notes have been assigned
    for group in gps_perf:
        for note in group:
            if not note.flag:
                perf_extras.append(note)
            
    return aligned_score, aligned_perf, score_extras, perf_extras


def distinct_times(notes_score, notes_perf):
    """
    Get list of start times of highest performed note from each distinct 
        score start time.  
    
    args:
        notes_score, notes_perf: aligned lists of pretty_midi note objects, 
                                 ordered by start time
    """
    if notes_score == []:
        return [], []
    
    times_score = [notes_score[0].start]
    times_perf= [notes_perf[0].start]
    for idx, note in enumerate(notes_score):
        time_score = note.start
        if time_score == times_score[-1]:
            # replace time to get time of highest note at this metrical position
            times_perf[-1] = notes_perf[idx].start
        elif time_score > times_score[-1]:
            # new metrical position
            times_perf.append(notes_perf[idx].start)
            times_score.append(time_score)
    
    return times_score, times_perf


def extrapolation_func(times_score, times_perf):
    """
    Get function for in-beat extrapolating

    times_score, times_perf: aligned lists of note start times, 
                             correspnding to highest note in each
                             metrical position of beats in the score
    """
    time_len = len(times_score)
    # set interpolation type
    kind = None
    if time_len == 2:
        kind = 'linear'
    elif time_len == 3:
        kind = 'quadratic'
    else:
        kind = 'cubic'
                    
    # extrapolation function
    f = interp1d(times_score, 
                 times_perf, 
                 kind=kind, 
                 fill_value='extrapolate', 
                 assume_sorted=True)
    
    return f


def extrapolate_beat(notes_score_gp, 
                     notes_perf_gp, 
                     beats_score, 
                     beats_perf, 
                     cubic_len=6,
                     in_beat_weight=1):
    """
    Calculates the expected beat times by a weighted combination of cubic 
        extrapolation from previous beats and linear extrapolation from within the beat

    args:
        notes_score_gp: score pretty_midi note objects, grouped by beat
        notes_perf_gp: performance pretty_midi note objects, grouped by beat
        beats_score: list of score beat times
        beats_perf: list of performance beat times
        cubic_len: number of beats considered for cubic spline interpolation function
        in_beat_weight: how much to weight to give in_beat extrapolation values, as opposed to
                        beat-to-beat extrapolation
    """
    # --- cubic beat extrapolation (beat continuity) --- #
    extrap = []
    x_range = np.arange(len(beats_perf))
    for idx, time in enumerate(beats_perf):
        if idx < 3:
            extrap.append(time)
        else:
            pts = slice(idx - min(idx, cubic_len), idx + 1)
            x = x_range[pts]
            y = beats_perf[pts]
            f = interp1d(x, y, kind='cubic', fill_value='extrapolate', assume_sorted=True)
            extrap.append(f(idx))
    
    # --- within-beat interpolation (local timing) --- #
    # extent beats for anacrusis
    beats_score_ext = [2 * beats_score[0] - beats_score[1]] + beats_score
    beats_perf_ext = [2 * beats_perf[0] - beats_perf[1]] + beats_perf

    in_beat = []
    # loop through beat groups (extrapolating for following beat time)
    for idx, gp in enumerate(notes_score_gp):
        # can't extrapolate from final group (no more beats)
        if idx == len(notes_score_gp) - 1:
            break
        
        # can't extrapolate from no notes
        elif gp == [] or notes_score_gp[idx + 1] == []:
            in_beat.append(extrap[idx])
        
        # only calculate if there are notes on the beat
        elif round(Decimal(notes_score_gp[idx + 1][0].start), 2) == round(Decimal(beats_score[idx]), 2):
            times_score, times_perf = distinct_times(gp, notes_perf_gp[idx])

            # add beat time if it is missing
            if round(Decimal(times_score[0]), 2) != round(Decimal(beats_score_ext[idx]), 2):
                times_score = [beats_score_ext[idx]] + times_score
                times_perf = [beats_perf_ext[idx]] + times_perf
            
            time_len = len(times_score)
            if time_len == 1:
                in_beat.append(extrap[idx])
            
            else:
                # extrapolate
                f = extrapolation_func(times_score, times_perf)
                in_beat.append(f(beats_score[idx]))
        
        else:
            in_beat.append(extrap[idx])
    
    # weighted average of extraplation types
    return [(ex + in_beat_weight * in_beat[i]) / (1 + in_beat_weight) 
            for i, ex in enumerate(extrap)]
    

def midi_divisions_per_beat(time_sigs):
    """
    args:
        time_sigs: list of time signature tokens
    return:
        list of (time, num), where num is the number of MIDI beats per 
            ASAP beat annotation position
    """
    reference = {
        '4/4': 4, '2/4': 2, '3/4': 3, '6/8': 2, '3/8': 3, '2/2': 2, '12/8': 4,
        '6/4': 2,'12/16': 4,'9/8': 3, '4/8': 4, '5/4': 5, '4/2': 4, '3/2': 3,
        '5/8': 5, '1/2': 1, '24/16': 8, '6/16': 2, '1/4': 1
    }
        
    beat_skips_out = []
        
    for time, ts_str in time_sigs:
        num_divs = int(re.search(r'\d+', ts_str).group())
        beat_skips_out.append((time, num_divs / reference[ts_str]))
        
    return beat_skips_out


