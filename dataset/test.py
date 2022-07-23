"""
Testing for dataset module
"""

import unittest

import utils
import token_funcs
import asap_metrics
import os
import sys
import itertools

import pretty_midi as pm


def pm_seq():
    time_sigs = ((6, 8, 1.1), (4, 4, 2), (12, 8, 3.28))
    time_sigs = [pm.TimeSignature(*ts) for ts in time_sigs]
    pmid = pm.PrettyMIDI()
    pmid.time_signature_changes = time_sigs
    return pmid


class Utils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_path = 'Bach/Fugue/bwv_846/Shi05M.mid'
        cls.metadata = utils.get_metadata()
        cls.pmid = pm_seq()
    
    def test_get_metadata(self):
        self.assertIsInstance(Utils.metadata, dict)
        self.assertEqual(list(Utils.metadata.keys())[0], Utils.test_path)
        self.assertEqual(len(Utils.metadata), 1067)

    def test_get_time_sigs(self):
        exp = [(1.1, '6/8'), (2, '4/4'), (3.28, '12/8')]
        self.assertEqual(utils.get_time_sigs(Utils.pmid), exp)
    
    def test_quantize(self):
        factors = [0.4, 0.12, 2]
        vals = [3.6, 3.71, 3.78, 10.83]
        exp = [[3.6, 3.6, 3.6, 10.8], [3.6, 3.72, 3.84, 10.8], [4, 4, 4, 10]]
        for idx, factor in enumerate(factors):
            for i, val in enumerate(vals):
                self.assertEqual(utils.quantize(val, factor), exp[idx][i])
    
    def test_bound(self):
        bds = (1, 2.3)
        vals = [-1.2, 1, 2.29, 2.31]
        exp = [1, 1, 2.29, 2.3]
        for idx, val in enumerate(vals):
            self.assertEqual(utils.bound(val, bds), exp[idx])
    
    def test_Band(self):
        band1 = utils.Band(bounds=[-1, 2, 3, 5, 6], hysteresis=(0, 0.3))
        band2 = utils.Band(initial_value=2.9, bounds=[-1, 2, 3, 5, 6], 
                           hysteresis=(0.3, 0), upper_bound=True)
        with self.assertRaises(RuntimeError):
            band1.update(-2)
        vals = []
        for x in (3.1, 5.1, 5.4, 2.1, 5.1):
            vals.append(band1.update(x))
        vals.append(band1.new_band(6.1))
        vals.append(band1.update(7.1))
        self.assertEqual(vals, [2, 2, 3, 1, 3, 4, 4])
        self.assertEqual(band2.band, 1)
        with self.assertRaises(RuntimeError):
            band2.update(6.1)

    def test_ibis(self):
        beats = [2, 3, 5, 6, 6.1]
        exp_ibis = [1, 2, 1, 0.1, 0.1]
        result = utils.ibis(beats)
        for i, ibi in enumerate(result):
            self.assertAlmostEqual(exp_ibis[i], ibi)
    
    def testTempoBand(self):
        band = utils.TempoBand()
        ibis = [0.5, 0.51, 0.53, 0.1]
        exp = [3, 3, 2, 4]
        res = []
        for ibi in ibis:
            res.append(band.update(ibi))
        self.assertEqual(res, exp)
        
    def test_group_by_beat(self):
        beats = (1, 2, 4, 5)
        times, pitches = (0.5, 1, 1.5, 7.1, 7.1, 7.2), (1, 1, 1, 1, 2, 1)
        tokens = [(time, pitches[i]) for i, time in enumerate(times)]
        notes = [pm.Note(0, pitches[i], time, time + 1) for i, time in enumerate(times)]
        aligned_times = (4.1, 1.2, 3, 3, 2, 5)
        aligned = [(time, pitches[i]) for i, time in enumerate(aligned_times)]
        gps, aligned_gps = utils.group_by_beat(beats, tokens=tokens, 
                                               aligned_tokens=aligned)
        a_notes = [pm.Note(0, pitches[i], time, time + 1) 
                   for i, time in enumerate(aligned_times)]
        self.assertEqual(gps, [[(0.5, 1)], [(1, 1), (1.5, 1)], [], [], 
                               [(7.1, 1), (7.1, 2), (7.2, 1)]])
        self.assertEqual(aligned_gps, [[(4.1, 1)], [(1.2, 1), (3, 1)], [], [], 
                                       [(3, 1), (2, 2), (5, 1)]])
        # test with note objects
        n_gps, n_aligned_gps = utils.group_by_beat(beats, notes=notes, 
                                                   aligned_notes=a_notes)
        self.assertEqual(n_gps, [[notes[0]], [notes[1], notes[2]], [], [], 
                                 [notes[3], notes[4], notes[5]]])
        self.assertEqual(n_aligned_gps, [[a_notes[0]], [a_notes[1], a_notes[2]], [], [], 
                                         [a_notes[3], a_notes[4], a_notes[5]]])
        # test adding flag to notes
        gps, a_gps = utils.group_by_beat(beats, tokens=tokens, add_flag=True,
                                         aligned_tokens=aligned)
        t1, t2 = tuple([*tokens[0], False]), tuple([*aligned[1], False])
        self.assertEqual((t1, t2), (gps[0][0], a_gps[1][0]))
    
    def test_align_notes(self):
        beats_score, beats_perf = [1, 2, 4, 6], [1, 5, 8, 10]
        t_p_score = ((0.5, 1), (0.5, 2), (0.6, 1), (1.1, 1), (2.0, 3), (7.0, 4), (7.0, 5))
        t_p_perf = ((0.6, 1), (0.7, 2), (0.9, 1), (1.2, 2), (5.2, 2), (11.1, 3), (11.1, 4))
        notes_score = [pm.Note(0, i[1], i[0], 1) for i in t_p_score]
        notes_perf = [pm.Note(0, i[1], i[0], 1) for i in t_p_perf]
        a_sco, a_per, sco_ex, per_ex = utils.align_notes(notes_score, notes_perf, 
                                                         beats_score, beats_perf)
        self.assertEqual(a_sco, [notes_score[i] for i in (0, 1, 3, 5)])
        self.assertEqual(a_per, [notes_perf[i] for i in (0, 1, 2, 6)])
        self.assertEqual(sco_ex, [notes_score[i] for i in (2, 4, 6)])
        self.assertEqual(per_ex, [notes_perf[i] for i in (3, 4, 5)])

    def test_distinct_times(self):
        times_score, times_perf = [1, 1, 1, 2, 3, 5, 5, 7], [1, 2, 2, 2, 3, 3, 4, 5]
        pitches = [1, 2, 3, 4, 5, 6, 7, 8]
        exp_idx = [2, 3, 4, 6, 7]
        notes_score, notes_perf = [], []
        for idx, pitch in enumerate(pitches):
            notes_score.append(pm.Note(0, pitch, times_score[idx], 1))
            notes_perf.append(pm.Note(0, pitch, times_perf[idx], 1))
        self.assertEqual(utils.distinct_times(notes_score, notes_perf), 
                    ([times_score[i] for i in exp_idx], [times_perf[i] for i in exp_idx]))

    def test_extrapolation_func(self):
        f = utils.extrapolation_func([1, 2], [2, 3])
        self.assertEqual(f(3), 4)
        f1 = utils.extrapolation_func([-1, 0, 1], [2, 0, 2])
        self.assertEqual(f1(3), 18)

    def test_extrapolate_beat(self):
        beats_s, beats_p = [1, 2, 3, 4], [1, 3, 4, 6]
        times_s = [0.5, 0.75, 1, 1.25, 1.25, 1.5, 1.75, 2, 3.5, 4]
        times_p = [0, 0.5, 1, 1.2, 1.505, 2.01, 2.515, 3, 4.98, 8]
        notes_s = [pm.Note(0, 1, time, 1) for time in times_s]
        notes_p = [pm.Note(0, 1, time, 1) for time in times_p]
        notes_s_gp, notes_p_gp = utils.group_by_beat(beats_s, notes=notes_s, 
                                                     aligned_notes=notes_p)
        extrap = utils.extrapolate_beat(notes_s_gp, notes_p_gp, beats_s, beats_p)
        for idx, val in enumerate([1, 3, 4, 6]): 
            self.assertAlmostEqual(extrap[idx], val, 1)
            if idx in (1, 4):
                self.assertNotEqual(extrap[idx], val)

    def test_midi_divisions_per_beat(self):
        tokens = [(1, '4/4'), (2, '6/8')]
        self.assertEqual(utils.midi_divisions_per_beat(tokens), [(1, 1), (2, 3)])


class Token_funcs(unittest.TestCase):
    
    # test_path = 'Bach/Fugue/bwv_846/Shi05M.mid'
    
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.metadata = get_metadata()
    
    @classmethod
    def setUpClass(cls):
        cls.test_path = 'Bach/Fugue/bwv_846/Shi05M.mid'
        cls.metadata = utils.get_metadata()

    def test_bar_tokens(self):
        beats, dbs = [1, 2, 3, 4, 5], [2, 4, 5]
        exp = [int(beat in dbs) for beat in beats]
        self.assertEqual(token_funcs.bar_tokens(beats, dbs), exp)
    
    def test_beat_tokens(self):
        beat_times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        exp_tokens = [3, 4, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5]
        downbeat_times = [3, 7, 10]
        tokens = token_funcs.beat_tokens(beat_times, downbeat_times)
        self.assertEqual(tokens, exp_tokens)

        # also check with a real example
        beat_times = Token_funcs.metadata[Token_funcs.test_path]['midi_score_beats']
        downbeat_times = Token_funcs.metadata[Token_funcs.test_path]['midi_score_downbeats']
        tokens = token_funcs.beat_tokens(beat_times, downbeat_times)
        self.assertEqual(len(beat_times), len(tokens))

    def test_tempo_tokens(self):
        beats = [1, 2, 3, 3.5, 4, 4.5, 5, 10, 11, 12, 13]
        exp_bands = [1, 1, 3, 3, 3, 3, 0, 1, 1, 1, 1]
        exp_medians = [60, 60, 120, 120, 60, 60, 60, 60, 60, 60, 60]
        _, b, l = token_funcs.tempo_tokens(beats)
        self.assertEqual(b, exp_bands)
        self.assertEqual(l, exp_medians)

    def test_time_sig_tokens(self):
        beats = [1, 2, 4, 5]
        ts = [(0.5, '4/4'), (3, '6/8'), (5, 'c')]
        exp = ['4/4', '4/4', '6/8', 'other']
        sigs = asap_metrics.time_signature(verbose=False)
        self.assertEqual(token_funcs.time_sig_tokens(beats, ts, sigs), exp)
        ts[0] = (1.5, '4/4')
        exp[1] = ('conti')
        self.assertEqual(token_funcs.time_sig_tokens(beats, ts, sigs, conti=True), exp)
        with self.assertRaises(RuntimeError):
            token_funcs.time_sig_tokens(beats, ts, sigs, allow_other=False)

    def test_note_tokens(self):
        beats = [1, 2, 4, 5]
        vals = ((1, 0.5, 0.6), (2, 0.5, 1.2), (2, 2, 4.5), (1, 2, 4), (3, 2, 3), (1, 5.1, 6.4))
        notes = [pm.Note(1, *n) for n in vals]
        exp_p = {0: [(0.5, 1), (0.5, 2)], 1: [], 2: [(2, 1), (2, 2), (2, 3)], 3: [], 
                 4: [(5.1, 1)]}
        exp_s = [[(0.5, 0.5), (0.5, 0.5)], [], [(2, 0), (2, 0), (2, 0)], [], [(5.1, 0.1)]]
        exp_d = [[(0.5, 0.1), (0.5, 0.7)], [], [(2, 1), (2, 1.5), (2, 0.5)], [], [(5.1, 0.9)]]
        exp_d_fu = [[(0.5, 0), (0.5, 0)], [], [(2, 1), (2, 1), (2, 0)], [], [(5.1, 0)]]
        exp_d_fr = [[(0.5, 0.1), (0.5, 0.7)], [], [(2, 0), (2, 0.5), (2, 0.5)], [], [(5.1, 0.9)]]
        p, s, d, d_fu, d_fr = token_funcs.note_tokens(notes, beats)
        self.assertEqual(p, exp_p)
        self.assertEqual(s, {i: e for i, e in enumerate(exp_s)})
        self.assertEqual(d, {i: e for i, e in enumerate(exp_d)})
        self.assertEqual(d_fu, {i: e for i, e in enumerate(exp_d_fu)})
        self.assertEqual(d_fr, {i: e for i, e in enumerate(exp_d_fr)})
    
    def test_dynamics_tokens(self):
        bands = [0, 31, 63, 95, 127]
        beats = [1, 2, 4, 5]
        times = [1, 1, 1, 1.5, 4.5, 4.6, 5.5]
        vals = [20, 32, 40, 80, 70, 60, 80]
        n = [pm.Note(v, 1, times[i], 1) for i, v in enumerate(vals)]
        notes = [[], [n[0], n[1], n[2], n[3]], [], [n[4], n[5]], [n[6]]]
        # band of local median (default win len = 1 beat)
        exp_loc_b = [1, 1, 1, 2]
        # local mean velocity (default window length 3, quant=1)
        exp_loc_m = [43, 50, 70, 70]
        # local standard deviation (default window lenth3, quant=1)
        exp_loc_std = [23, 21, 8, 8]
        exp_n_vel = {0: [], 1: [(1, 20), (1, 32), (1, 40), (1.5, 80)], 
                     2: [], 3: [(4.5, 70), (4.6, 60)], 4: [(5.5, 80)]}
        exp_n_vel_b = {0: [], 1: [(1, 0), (1, 0), (1, 1), (1.5, 2)], 
                       2: [], 3: [(4.5, 2), (4.6, 2)], 4: [(5.5, 2)]}
        exp_n_std = {0: [], 1: [(1, -1), (1, -0.4), (1, -0.2), (1.5, 1.6)], 
                     2: [], 3: [(4.5, 0), (4.6, -1.2)], 4: [(5.5, 1.2)]}
        loc_m, loc_std, loc_b, n_std, n_vel, n_vel_b, _ = \
            token_funcs.dynamics_tokens(notes, beats, bands=bands, mean_len=3)
        self.assertEqual(loc_m, exp_loc_m)
        self.assertEqual(loc_std, exp_loc_std)
        self.assertEqual(loc_b, exp_loc_b)
        self.assertEqual(n_std, exp_n_std)
        self.assertEqual(n_vel, exp_n_vel)
        self.assertEqual(n_vel_b, exp_n_vel_b)
    
    def test_timing_labels(self):
        beats_s, beats_p = [1, 2, 4, 5], [1, 3, 7, 9] 
        times_s, times_p = [1, 2, 2, 2, 3, 3.5, 5.5], [0.5, 3, 3, 5, 4.6, 6.2, 9.3]
        ends_s, ends_p, = [2, 3, 4.5, 5.5, 4.5, 5, 6], [2, 7.2, 8, 11, 7.8, 9, 9.8], 
        pitches = [1, 2, 3, 4, 5, 6, 7]
        rough_interp_end_p = [3, 5, 8, 10, 8, 9, 11]
        rough_dur_pred = [e - times_p[i] for i, e in enumerate(rough_interp_end_p)]
        a = [((ends_p[i] - times_p[i]) - p) / p for i, p in enumerate(rough_dur_pred)]
        a = [round(v, 1) for v in a]
        # dur_s, dur_p = [1, 0.5, 1.5, 2.5, 1, 1.25, 0.5], [0.75, 1.1, 1.5, 2, 1, 1.2, 0.25]
        dur_s_gp = {0: [], 1: [(1, 1)], 2: [(2, 0.5), (2, 1.5), (2, 2.5), (3, 1), (3.5, 1.25)], 
                    3: [], 4: [(5.5, 0.5)]}
        notes_s = [pm.Note(1, pitches[i], t, ends_s[i]) for i, t in enumerate(times_s)]
        notes_p = [pm.Note(1, pitches[i], t, ends_p[i]) for i, t in enumerate(times_p)]
        notes_s_gp = [[], [notes_s[0]], 
                      [notes_s[1], notes_s[2], notes_s[3], notes_s[4], notes_s[5]], 
                      [], [notes_s[6]]]
        notes_p_gp = [[], [notes_p[0]], 
                      [notes_p[1], notes_p[2], notes_p[3], notes_p[4], notes_p[5]], 
                      [], [notes_p[6]]]
        # a2 = [((ends_p[i] - times_p[i]) - (ends_s[i] - times_s[i]) * 2) 
        #      / ((ends_s[i] - times_s[i]) * 2) for i in range(len(pitches))]
        exp_art = {0: [], 1: [(1, a[0])], 2: [(2, a[1]), (2, a[2]), (2, a[3]), (3, a[4]), (3.5, a[5])], 
                   3: [], 4: [(5.5, a[6])]}
        exp_dev = {0: [], 1: [(1, -0.25)], 2: [(2, 0), (2, 0), (2, 0.5), (3, -0.1), (3.5, 0.32)], 
                   3: [], 4: [(5.5, -0.35)]}
        art, _, _, dev, _, _ = token_funcs.timing_labels(notes_s_gp, notes_p_gp, dur_s_gp, 
                                                         beats_s, beats_p, calc_type='dynamic',
                                                         artic_quant=0.1)
        for i, v in art.items():
            if v == []:
                self.assertEqual(exp_art[i], [])
            else:
                for idx, (time, val) in enumerate(v):
                    self.assertAlmostEqual(time, exp_art[i][idx][0], 2, f"beat {i}, num {idx}")
                    self.assertAlmostEqual(val, exp_art[i][idx][1], 2, f"beat {i}, num {idx}")
                    self.assertAlmostEqual(time, exp_dev[i][idx][0], 2, f"beat {i}, num {idx}")
                    self.assertAlmostEqual(dev[i][idx][1], exp_dev[i][idx][1], 2, 
                                           f"beat {i}, num {idx}") 
        art, _, _, dev, _, _ = token_funcs.timing_labels(notes_s_gp, notes_p_gp, dur_s_gp, 
                                                        beats_s, beats_p, calc_type='linear')
        exp_art = {0: [], 1: [(1, -0.25)], 2: [(2, 1.2), (2, 0), (2, 0), (3, 0), (3.5, -0.05)], 
                   3: [], 4: [(5.5, -0.5)]}
        exp_dev = {0: [], 1: [(1, -0.25)], 2: [(2, 0), (2, 0), (2, 0.5), (3, -0.1), (3.5, 0.05)], 
                   3: [], 4: [(5.5, -0.35)]}
        for i, v in art.items():
            if v == []:
                self.assertEqual(exp_art[i], [])
            else:
                for idx, (time, val) in enumerate(v):
                    self.assertEqual(time, exp_art[i][idx][0], f"beat {i}, num {idx}")
                    self.assertEqual(val, exp_art[i][idx][1], f"beat {i}, num {idx}")
                    self.assertEqual(time, exp_dev[i][idx][0], f"beat {i}, num {idx}")
                    self.assertEqual(dev[i][idx][1], exp_dev[i][idx][1], f"beat {i}, num {idx}")           
    
    def test_harmonic_tokens(self):
        path = os.path.join("dataset/asap-dataset/", Token_funcs.test_path)
        # path = os.path.join("asap-dataset/", Token_funcs.test_path)
        # mtk_score = miditoolkit.midi.parser.MidiFile(path, ticks_per_beat=384)
        score = pm.PrettyMIDI(path)
        # time_sigs = utils.get_time_sigs(score)
        # corpus_sigs = asap_metrics.time_signature(verbose=False)
        beats_score = Token_funcs.metadata[Token_funcs.test_path]['midi_score_beats']
        # time_sig_tokens = token_funcs.time_sig_tokens(beats_score, time_sigs, corpus_sigs)
        notes = list(itertools.chain(*[i.notes for i in score.instruments]))
        notes_score_gp, _ = utils.group_by_beat(beats_score, notes=notes)
        key, qual = token_funcs.harmonic_tokens(notes_score_gp, beats_score)
        print(f"\nHarm tokens: {key[:20]}\n")
        print(f"\nHarm quality tokens: {qual[:20]}\n")
    
    def test_rubato_tokens(self):
        pass


def print_timing_labels(track_num):
    metadata = utils.get_metadata()
    path = list(metadata.keys())[1]
    midi_perf_path = os.path.join('dataset/asap-dataset', path)
    midi_score_path = os.path.join(os.path.split(midi_perf_path)[0], 'midi_score.mid')
    perf = pm.PrettyMIDI(midi_perf_path)
    score = pm.PrettyMIDI(midi_score_path)
    score_notes = list(itertools.chain(*[i.notes for i in score.instruments]))
    perf_notes = list(itertools.chain(*[i.notes for i in perf.instruments]))
    beats_perf = metadata[path]['performance_beats']
    beats_score = metadata[path]['midi_score_beats']
    score_notes, perf_notes, _, _ = utils.align_notes(score_notes,
                                                    perf_notes,
                                                    beats_score,
                                                    beats_perf,
                                                    search_range=0.25)
    articulation, timing_dev = token_funcs.timing_labels(score_notes,
                                                        perf_notes,
                                                        None,
                                                        beats_score,
                                                        beats_perf)
    print("Articulation: ")
    for i, vals in articulation.items():
        print(f"Beat {i}: {[t[1] for t in vals]}")
    print("Timing: ")
    for i, vals in timing_dev.items():
        print(f"Beat {i}: {[t[1] for t in vals]}")   


if __name__ == "__main__":

    if len(sys.argv) == 1:
        unittest.main()

    elif sys.argv[1] == 'print_timing_labels':
        print_timing_labels(int(sys.argv[2]))

