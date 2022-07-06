"""
References:
    https://github.com/fosfrancesco/asap-dataset
    https://github.com/YatingMusic/remi/blob/master/midi2remi.ipynb
    https://web.mit.edu/music21/doc/moduleReference/moduleAnalysisFloatingKey.html
    https://github.com/joshuachang2311/chorder
    https://github.com/YatingMusic/miditoolkit
    https://github.com/YatingMusic/compound-word-transformer/blob/main/dataset/analyzer.py

Token types:
    
    ===== score input and expression output tokens =====

        type (metatoken for compound words)
        
        bar
        
        beats (numbered)
        
        inter-beat-interval
        tempo band (coarse)
        local tempo (smoothed)

        time signature

        note pitch
        note start time (offset from beat as proportion of beat)
        note duration (proportion of beat)
        
        local dynamics mean (metric token)
        local dynamics standard deviation (metric token) 
        local dynamics band (metric token)
        note velocity standard deviations from local mean
        note absolute velocity
        note velocity band
        
        note articulation label (note length relative to score as proportion of IBI)
        note expressive timing deviation (difference from expected note start time, 
                                          as proportion of IBI)
        
        key (metric token)
        harmonic quality (metric token)
    
    ===== attribute tokens =====
        
        rubato

"""

from decimal import Decimal
import statistics
import collections
import functools

import pretty_midi
import miditoolkit
from chorder import Dechorder
from scipy.interpolate import interp1d

import utils


def type_token_val(type_str):
    if type_str == 'eos':
        return 1
    elif type_str == 'metric':
        return 2
    elif type_str == 'note':
        return 3
    else:
        raise RuntimeError(f"Invalid string for type token: {type_str}")


def bar_tokens(beats, downbeats):
    """
    args:
        beats: list of beat times
        downbeats: list of downbeat times
    returns:
        downbeat_tokens: (time, 1)
    """
    bar_out = []
    db_it = iter(downbeats + ['dummy'])
    db = next(db_it)
    for time in beats:
        if time == db:
            val = 1
            db = next(db_it)
        else: 
           val = 0
        bar_out.append((time, val))
    return bar_out


def beat_tokens(beats, downbeats):
    """
    Given beat and downbeat times in a MIDI score/performance,
        return list of tuples of (time, beat_number)
    """
    beats_out = []
    
    # add dummy value to end of downbeat_times
    downbeats.append(0)
    
    # for any beats before first downbeat
    leading_beats = 0
    # to find initial time signature
    initial_BPB = 0
    
    # find number of beats in anacrusis
    for time in beats:
        if time < downbeats[0]:
            leading_beats += 1
        # find number of beats in first bar
        elif time < downbeats[1] and leading_beats > 0:
            initial_BPB += 1
        else:
            break
    
    # downbeat_times iterator
    db_it = iter(downbeats + ['dummy'])
    db = next(db_it)
    beat_num = 1
    
    for time in beats:
        # deal with beats before first downbeat
        if leading_beats > 0:
            beat = initial_BPB - leading_beats + 1
            if beat <= 0:
                raise RuntimeError("More initial beats than time signature allows")
            beats_out.append((time, beat))
            leading_beats -= 1
        
        else:
            # check if we have reached a downbeat
            if time == db:
                beat_num = 1
                db = next(db_it)

            # record beat token
            beats_out.append((time, beat_num))
            beat_num += 1
    
    return beats_out


def tempo_tokens(beats, 
                 ibi_tokens=False,
                 band_tokens=True,
                 lower_bounds=[5, 45, 85, 120, 150],
                 hysteresis=[5, 5],
                 allow_zero=False,
                 local_tempo_tokens=True,
                 median_time=4):
    """
    Inter-beat intervals (IBIs): length of time in secs between successive beats
    
    Tempo band: Quantized IBIs
        Default bands (with default 5 BPM hysteresis):
            adagissimo: <45 BPM
            lento: 45 - 85 BPM
            moderato: 85 - 120 BPM
            allegro: 120 - 150 BPM
            presto: >150 BPM
        
    Local tempo: median IBI of b beats, centred on the label position
                 b depends on the tempo class, to roughly equal 4 seconds 
                 of playing time:
                    adagissimo: b = 1
                    lento: b = 5
                    moderato: b = 7
                    allegro: b = 9
                    presto: b = 11
    """
    ibi_vals = utils.ibis(beats)
    # ibi tokens
    ibi_out = None
    if ibi_tokens:
        ibi_out = []
        for i, time in enumerate(beats):
            ibi_out.append((time, ibi_vals[i]))
    
    band_out = None
    local_tempo_out = None
    # tempo band tokens
    if band_tokens or local_tempo_tokens:
        band_out = []
        band = utils.TempoBand(tempo=ibi_vals[0], 
                               lower_bounds=lower_bounds,
                               hysteresis=hysteresis,
                               allow_zero=allow_zero)
        
        for i, ibi in enumerate(ibi_vals):
            band_out.append((beats[i], band.update(ibi)))
        
        # local tempo tokens
        if local_tempo_tokens:
            local_tempo_out = []

            # calculate median window length
            win_lens = []
            for i, bound in enumerate(lower_bounds):
                if i < len(lower_bounds) - 1:
                    centre = (bound + lower_bounds[i + 1]) / 2
                else:
                    centre = bound + 20
                win_len = median_time / (60 / centre)
                
                # store length of window either side of central value
                win_lens.append(max(1, int(win_len // 2)))
            
            # calculate local tempi
            # median filter IBIs, adjusting filter window by tempo band
            for i, (time, band) in enumerate(band_out):
                win_len = min(win_lens[band], i, len(ibi_vals) - i - 1)
                vals = ibi_vals[i - win_len: i + win_len + 1]
                local_tempo_out.append((time, 60 / statistics.median(vals)))
    
    return ibi_out, band_out, local_tempo_out


def time_sig_tokens(beats,
                    time_sigs, 
                    corpus_sigs,
                    allowed_time_sigs=None, 
                    allow_other=True, 
                    min_freq=0,
                    conti=False):
    """
    args:
        time_sigs: list of (time, 'time_sig')
        allowed time_sigs: list of allowed time_sigs
        allow_other: if True, any time signature not in allowed_time_sigs
                     will be assigned class 'other'
        min_freq: cutoff minimum allowed frequency for time signature
                  in ASAP dataset
        conti: add 'continuation tokens' at beats where there is no time_sig
               change (rather than the current time_sig)
    returns:
        list of tuples (time, time_sig_class)
    """
    # track position in time_sigs sequence
    ts = None
    ts_ptr = 0

    if allowed_time_sigs is None:
        allowed_time_sigs = [sig[0] for sig in corpus_sigs if sig[3] >= min_freq]
    
    time_sig_out = []
    # loop through beats
    for time in beats:
        # if we have a new time_sig
        if not ts_ptr or (ts_ptr < len(time_sigs) and time_sigs[ts_ptr][0] <= time):
            # move on current time_sig
            ts = time_sigs[ts_ptr][1]
            # add token
            if ts in allowed_time_sigs:
                time_sig_out.append((time, ts))
            elif allow_other:
                time_sig_out.append((time, 'other'))
            else:
                raise RuntimeError(f"Illegal time signature provided: {ts}")
            # move on pointer
            ts_ptr += 1
        elif conti:
            time_sig_out.append((time, 'conti'))
        else:
            time_sig_out.append((time, ts))
    
    return time_sig_out


def note_tokens(notes,
                beats_score,
                duration_quant=1/120,
                start_quant=1/120,
                split_duration=True):
    """
    args:
        notes: list of pretty_midi note objects
        beats_score: list of beat times for score (absolute, secs)
        start_quant: quantization factor for start time relative to beat
        duration_quant: list of possible note duration values (as proportion of a beat)
    return:
        note_pitch
        note_start
        note_duration
        note_duration (full beats)
        note_duration (fraction part)
    """
    # extend beats to allow calculations relating to final beat
    ext_beats = beats_score + [beats_score[-1] + beats_score[-1] - beats_score[-2]]
    # extend beats to allow for anacrusis
    ext_beats = [2 * beats_score[0] - beats_score[1]] + ext_beats
    
    # initialise outputs
    note_pitch_out = {i: [] for i in range(len(beats_score) + 1)}
    note_start_out = {i: [] for i in range(len(beats_score) + 1)}
    note_dur_full_out = None
    note_dur_fract_out = None
    if split_duration:
        note_dur_full_out = {i: [] for i in range(len(beats_score) + 1)}
        note_dur_fract_out = {i: [] for i in range(len(beats_score) + 1)}
    note_dur_out = {i: [] for i in range(len(beats_score) + 1)}
    
    # group notes by beat
    notes_gp, _ = utils.group_by_beat(beats_score, notes=notes)

    for idx, gp in enumerate(notes_gp):
        for note in gp:
            # note pitch tokens
            note_pitch_out[idx].append((note.start, note.pitch))
            
            # find ratio of start position to beat length    
            start_ratio = (note.start - ext_beats[idx]) / \
                        (ext_beats[idx + 1] - ext_beats[idx])
            
            # quantize ratio
            note_start_out[idx].append((note.start, utils.quantize(start_ratio, 
                                                                   start_quant)))
            
            # note duration tokens (proportion of a beat)
            # find duration of note relative to beats
            duration = 0
            # deal with case where note lasts into next beat
            if note.end > ext_beats[idx + 1]:
                # whole note part
                while idx < len(ext_beats) - duration - 2 and \
                    note.end > ext_beats[idx + 2 + duration]:
                    duration += 1

                # fraction of last beat
                try:
                    duration += (note.end - ext_beats[idx + int(duration) + 1]) / \
                                (ext_beats[idx + int(duration) + 2] - 
                                 ext_beats[idx + int(duration) + 1])
                except IndexError:
                    duration += 0

                # fraction of first beat
                duration += (ext_beats[idx + 1] - note.start) / \
                            (ext_beats[idx + 1] - ext_beats[idx])
                
            else:
                duration = (note.end - note.start) / \
                        (ext_beats[idx + 1] - ext_beats[idx])
            
            # duration output(s)
            if split_duration:
                note_dur_full_out[idx].append((note.start, int(duration)))
                note_dur_fract_out[idx].append((note.start, 
                                                utils.quantize(duration % 1, duration_quant)))

            note_dur_out[idx].append((note.start, 
                                      utils.quantize(duration, duration_quant)))
    
    return note_pitch_out, note_start_out, note_dur_out, \
           note_dur_full_out, note_dur_fract_out


def dynamics_tokens(notes_gp, 
                    beats,
                    mean_len=3,
                    mean_quant=1,
                    std_quant=1,
                    bands=[],
                    band_win=1,
                    band_hysteresis=(5, 5),
                    note_std_quant=0.2,
                    note_std_bounds=(-15, 15)):
    """
    args:
        notes_gp: list of pretty_midi note objects, grouped by beat
        beats: list of beat times
        mean_len: number of bars for mean calculation
        mean_quant: size of discrete bins for mean tokens
        std_quant: size of discrete bins for standard deviation tokens
        bands: bounds of velocity bands
        band_win: number of bars for median calculation for local velocity band
        band_hysteresis: amount of hysteresis (low, high) for velocity bands
        note_std_quant: quantization for standard deviation relative to local mean
        note_std_bounds: restricts output to(min, max) number of standard deviations 
                         from local mean

    returns:
        local_velocity_band_tokens
        local_velocity_mean_tokens
        local_velocity_std_tokens
        note_absolute_velocity_tokens
        note_velocity_band_tokens
        note_velocity_std_tokens
    """
    # # group notes by beat
    # notes_gp, _ = utils.group_by_beat(beats, notes=pm_notes)
    
    # extend beats list for anacrusis
    ext_beats = [0] + beats

    # for calculating bands
    local_band = utils.Band(bounds=bands, upper_bound=True, hysteresis=band_hysteresis)
    note_band = utils.Band(bounds=bands, upper_bound=True, hysteresis=band_hysteresis)
    
    local_mean_out, local_std_out, local_band_out = [], [], []
    note_std_out = {i: [] for i in range(len(beats) + 1)}
    note_abs_vel_out = {i: [] for i in range(len(beats) + 1)}
    note_band_out = {i: [] for i in range(len(beats) + 1)}
    num_gps = len(notes_gp)
    prev_mean = 0
    prev_median = 0
    for idx, group in enumerate(notes_gp):
        # no metic token for anacrusis
        if idx > 0:
            # local mean velocity
            gp_range = (max(idx - mean_len // 2, 0), min((idx + mean_len // 2) + 1, num_gps))
            gp_slice = functools.reduce(lambda x, y: x + y, notes_gp[slice(*gp_range)])
            velocities = [n.velocity for n in gp_slice]
            # if no notes in range
            if not len(velocities):
                mean_vel = prev_mean
            else:
                mean_vel = statistics.mean(velocities)
                prev_mean = mean_vel
            
            local_mean_out.append((ext_beats[idx], utils.quantize(mean_vel, mean_quant)))
            
            # local velocity standard deviation
            std_vel = statistics.pstdev(velocities, mu=mean_vel)
            local_std_out.append((ext_beats[idx], utils.quantize(std_vel, std_quant)))
            
            # local velocity band
            gp_range = (max(idx - band_win // 2, 0), min((idx + band_win // 2) + 1, num_gps))
            gp_slice = functools.reduce(lambda x, y: x + y, notes_gp[slice(*gp_range)])
            velocities = [n.velocity for n in gp_slice]
            # if no notes in range
            if not len(velocities):
                median_vel = prev_median
            else:
                median_vel = statistics.median(velocities)
                prev_median = median_vel
            local_band_out.append((ext_beats[idx], local_band.update(median_vel)))
        
        for note in group:
            # note-wise deviation from local velocity mean
            dev = utils.quantize((note.velocity - mean_vel) / std_vel, note_std_quant)

            if note_std_bounds is not None:
                dev = utils.bound(dev, note_std_bounds)
            
            note_std_out[idx].append((note.start, dev))
            
            # absolute velocity
            note_abs_vel_out[idx].append((note.start, note.velocity))
            
            # velocity band
            note_band_out[idx].append((note.start, note_band.update(note.velocity)))
    
    return local_mean_out, local_std_out, local_band_out, \
            note_std_out, note_abs_vel_out, note_band_out
            

def timing_labels(notes_score,
                  notes_perf,
                  dur_score_gp,
                  beats_score,
                  beats_perf,
                  artic_quant=0.1,
                  artic_lims=None,
                  dev_quant=0.01, 
                  dev_lims=None,
                  cubic_len=6,
                  beat_in_beat_weight=1,
                  non_beat_in_beat_weight=2):
    """
    args:
        notes_score: notes from score, aligned by list position with notes_perf
        notes_perf: notes from performed version, aligned with score notes
        dur_score_gp: list of duration tokens for score notes
        beats_score: list of beat times for score (absolute, secs)
        beats_perf: list of beats times for performed version
        artic_quant: quantization size (proportion of a beat) for articulation values
        artic_lims: (lower, upper) bounds for articulation value
        dev_quant: resolution (proportion of a beat) for timing deviation labels
        dev_lims: (lower, upper) bounds for timing deviation value
        cubic_len: number of beats considered for cubic spline interpolation function
        beat_in_beat_weight: how much to weight to give in_beat extrapolation values, 
                             as opposed to beat-to-beat extrapolation for notes 
                             corresponding with score beats
        non_beat_in_beat_weight: how much to weight to give in_beat extrapolation values, 
                                 as opposed to beat-to-beat extrapolation for notes not
                                 corresponding with score beats
    return:
        note_articulation_labels: tokens for note articulation deviation from expected
                                  (unit: proportion of a beat)
        note_deviation_labels: tokens for note timing deviation from expected
                               (unit: proportion of a beat)
    """
    # group by beat
    notes_score_gp, notes_perf_gp = utils.group_by_beat(beats_score, notes=notes_score, 
                                                        aligned_notes=notes_perf)
    # dur_score_gp, _ = utils.group_by_beat(beats_score, tokens=durations_score)
    
    # get IBIs
    ibis_perf = utils.ibis(beats_perf)
    # add IBI value for beat 0
    ibis_perf = [ibis_perf[0]] + ibis_perf

    # extend beats for beat 0 and n + 1
    beats_score_ext = [2 * beats_score[0] - beats_score[1]] + beats_score \
                      + [2 * beats_score[-1] - beats_score[-2]]
    beats_perf_ext = [beats_perf[0] - ibis_perf[0]] + beats_perf + \
                     [beats_perf[-1] + ibis_perf[-1]]
    
    # get expected beat times
    exp_beats = utils.extrapolate_beat(notes_score_gp, notes_perf_gp,
                                       beats_score, beats_perf,
                                       cubic_len=cubic_len,
                                       in_beat_weight=beat_in_beat_weight)
    
    # get cubic spline interpolation function for instantaneous timing
    beat_interp = interp1d(beats_score_ext, beats_perf_ext, kind='cubic',
                           fill_value='extrapolate', assume_sorted=True)
    
    artic_out = collections.defaultdict(list)
    dev_out = collections.defaultdict(list)
    for beat, notes_perf in enumerate(notes_perf_gp):

        for idx, note in enumerate(notes_perf):
            # --- articulation --- #
            # # predicted end time (using IBIs)
            # dur_pred = 0
            # note_dur_score = dur_score_gp[beat][idx][1]
            # beat_count = 0
            
            # # need to add first fractional part
            
            # # note: could just use interpolation function for predicted end point
            # # or both interpolations!!
            
            # # whole beats
            # while note_dur_score >= 1:
            #     dur_pred += ibis_perf[beat + beat_count]
            #     beat_count += 1
            #     note_dur_score -= 1
            
            # # end fractional part
            # dur_pred += ibis_perf[beat + beat_count] * note_dur_score

            # predicted end time via interpolation
            end_pred = beat_interp(notes_score_gp[beat][idx].end)
            dur_pred = end_pred - note.start
            
            # Note: could use:
            # predicted end time via score duration (proportion of performance IBIs)
            # OR: predicted end time relative to other notes
            
            # actual duration
            dur_perf = note.end - note.start
            
            # articulation difference as proportion of expected
            artic = (dur_perf - dur_pred) / dur_pred
            if artic_lims is not None:
                artic = artic_lims[0] if artic < artic_lims[0] else artic
                artic = artic_lims[1] if artic > artic_lims[1] else artic
            
            # get equivalent score time for reference
            time = notes_score_gp[beat][idx].start
            
            # add to output
            artic_out[beat].append((time, utils.quantize(artic, artic_quant)))

            # --- timing deviation --- #
            pred = 0
            # notes whose score position corresponds with a beat
            if beat > 1 and (round(Decimal(notes_score_gp[beat][idx].start), 2) == 
                             round(Decimal(beats_score_ext[beat]), 2)):
                pred = exp_beats[beat - 1]
            
            # in-beat notes
            else:
                # deviation relative to beat level curve
                # score_note = notes_score_gp[beat][idx]
                # score_time = (score_note.start - beats_score_ext[beat]) / \
                #     (beats_score_ext[beat + 1] - beats_score_ext[beat])
                # pred_time = beat_interp(score_time)

                pred_time = beat_interp(notes_score_gp[beat][idx].start)
                
                # deviation relative to in-beat curve
                in_beat_time = 0
                if idx in [0, 1]:
                    in_beat_time = pred_time
                
                else:
                    # get distinct note times
                    times_score, times_perf = utils.distinct_times(notes_score_gp[beat][:idx], 
                                                                notes_perf[:idx])
                        
                    # add beat time if it is missing
                    if times_score == [] or (round(Decimal(times_score[0]), 2) != 
                                            round(Decimal(beats_score_ext[beat]), 2)):
                        times_score = [beats_score_ext[beat]] + times_score
                        times_perf = [beats_perf_ext[beat]] + times_perf
                    
                    # remove last time if it corresponds with extrapolation time
                    if times_score[-1] == notes_score_gp[beat][idx]:
                        times_score.pop()
                        times_perf.pop()
                    
                    if len(times_score) < 2:
                        in_beat_time = pred_time
                    
                    else: 
                        f = utils.extrapolation_func(times_score, times_perf)
                        in_beat_time = f(notes_score_gp[beat][idx].start)
                
                # weighted average of predicted times
                pred = (pred_time + in_beat_time * non_beat_in_beat_weight) / \
                       (1 + non_beat_in_beat_weight)
                
            # get deviation token
            dev = (note.start - pred) / ibis_perf[beat]
            if dev_lims is not None:
                dev = dev_lims[0] if dev < dev_lims[0] else dev
                dev = dev_lims[1] if dev > dev_lims[1] else dev
            dev_out[beat].append((time, utils.quantize(dev, dev_quant)))

    return artic_out, dev_out


# def harmonic_tokens(mtk_score, time_sig_tokens, beats_score, 
#                     reduce=False, extra_reduce=False, conti=True):
def harmonic_tokens(notes_score_gp, beats_score, 
                    reduce=False, extra_reduce=False, conti=True):
    """
    args:
        mtk_score: MIDI score as a midi_toolkit object
    return:
        keys_tokens
        chord_quality_tokens
    """
    # # skip factor for each time_sig
    # skips = utils.midi_divisions_per_beat(time_sig_tokens)
    # skip = skips[0][1]
    # skips = {time: int(skip) for (time, skip) in skips}
    
    # # chords for each MIDI beat
    # chords = Dechorder.dechord(mtk_score)
    
    # # locate chord for first beat
    # tpb = mtk_score.ticks_per_beat 
    # tick_times = mtk_score.get_tick_to_time_mapping()
    # beat_ptr = -1
    # for i in range(int(len(tick_times) // tpb)):
    #     if round(Decimal(tick_times[i * tpb]), 2) == round(Decimal(beats_score[0]), 2):
    #         beat_ptr = i
    #         break
    # if beat_ptr == -1:
    #     raise RuntimeError("Annotation times do not match MIDI beat times.\n")
    
    # # chords tokens (per annotated beat)
    # keys_out = []
    # chord_quality_out = []
    # no_chord = '_'
    # no_change = 'same'
    # for beat in beats_score:
    #     if beat in skips:
    #         skip = skips[beat]

    #     chord = chords[beat_ptr]

    #     # reduce quality dimension
    #     if reduce:
    #         if chord.quality in ['o7', '/o7']:
    #             chord.quality = 'o'
    #         elif chord.quality in ['M7', 'sus2']:
    #             chord.quality = 'M'
    #         elif chord.quality == 'm7':
    #             chord.quality == 'm'
    #         if extra_reduce and chord.quailty in ['7', 'sus4']:
    #             chord.quality = 'M'
        
    #     # add to outputs
    #     if not chord.is_complete():
    #         keys_out.append((beat, no_chord))
    #         chord_quality_out.append((beat, no_change))
        
    #     elif beat > beats_score[0] and chord.root() == keys_out[-1][1]:
    #         keys_out.append((beat, no_change))
    #         chord_quality_out.append((beat, no_change))
        
    #     else: 
    #         keys_out.append((beat, chord.root()))
    #         chord_quality_out.append((beat, chord.quality))
        
    #     beat_ptr += skip
    
    beats_score_ext = beats_score + [2 * beats_score[-1] - beats_score[-2]]

    # group notes into beats
    
    keys_out = []
    chord_quality_out = []
    NO_CHORD = 'none'
    CONTI = 'conti'
    current_chord = NO_CHORD
    for idx, beat in enumerate(beats_score):
        # get notes
        notes = []
        for note in notes_score_gp[idx + 1]:
            start = int(round(480 * ((note.start - beat) / 
                                     (beats_score_ext[idx + 1] - beat))))
            end = int(round(480 * ((note.end - beat) / (beats_score_ext[idx + 1] - beat))))
            notes.append(miditoolkit.midi.containers.Note(note.velocity, note.pitch, 
                                                          start, end))
        
        # get chord
        if notes != []:
            chord = Dechorder.get_chord_quality(notes, notes[0].start, notes[-1].end, 
                                                consider_bass=True)
            chord = chord[0]
            
            # reduce quality dimension
            if reduce:
                if chord.quality in ['o7', '/o7']:
                    chord.quality = 'o'
                elif chord.quality in ['M7', 'sus2']:
                    chord.quality = 'M'
                elif chord.quality == 'm7':
                    chord.quality == 'm'
                if extra_reduce and chord.quailty in ['7', 'sus4']:
                    chord.quality = 'M'
            
            # add to outputs
            if not chord.is_complete():
                keys_out.append((beat, NO_CHORD))
                chord_quality_out.append((beat, NO_CHORD))
            
            elif conti and idx > 0 and chord.root() == keys_out[-1][1]:
                keys_out.append((beat, CONTI))
                chord_quality_out.append((beat, CONTI))
            
            else: 
                keys_out.append((beat, chord.root()))
                chord_quality_out.append((beat, chord.quality))
        
        else:
            keys_out.append((beat, NO_CHORD))
            chord_quality_out.append((beat, NO_CHORD))
    
    return keys_out, chord_quality_out


# === Attribute tokens === #

def rubato_tokens(beats, beats_type):
    """
    Return whether or not beats are rubato
    """
    rubato_out = []
    for beat in beats:
        rubato_out.append((beat, int(beats_type[beat] == 'bR')))
            
    return rubato_out



if __name__ == '__main__':
    test_path = './asap-dataset/Bach/Fugue/bwv_846/midi_score.mid'
    midi = miditoolkit.midi.parser.MidiFile(test_path)
    print(*midi.instruments[0].notes[:20], sep='\n')
    print(*midi.tempo_changes[:10], sep='\n')
    print()

    test_path_expressive = './asap-dataset/Bach/Fugue/bwv_846/Shi05M.mid'
    midi_exp = miditoolkit.midi.parser.MidiFile(test_path_expressive)
    print(*midi_exp.instruments[0].notes[:20], sep='\n')
    print(*midi_exp.tempo_changes[:10], sep='\n')

    test_piece = pretty_midi.PrettyMIDI(test_path_expressive)
    print(test_piece)
    
    times = midi.get_tick_to_time_mapping()
    print(f"Times fo ticks: {times}")


