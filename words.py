"""
Create Datasets of sequences of compound words - record parameters
"""
import sys
import os

from dataset.make_words import compute_words

params = {
    'baseline_t1': {
        'paths': ['dataset/tokens/t1', 'saves'],
        'kwargs': {
            'bar_tokens': None,
            'eos_tokens': (True, True),
            'type_tokens': True,
            'in_metric_t_types': ['beat', 'tempo_band'],
            'in_note_t_types': ['pitch', 'start', 'dur_full', 'dur_fract'],
            'attr_metric_t_types': [],
            'attr_note_t_types': [],
            'out_metric_t_types': ['local_tempo', 'local_vel_mean'],
            'out_note_t_types': ['articulation', 'timing_dev', 'note_rel_vel']
        }
    }
}


if __name__ == '__main__':
    """
    args:
        name: name of folder, matching params dictionary key above
    """
    # get name from command line
    name = sys.argv[1]

    words_path = os.path.join(params[name]['paths'][1], name + '/words')

    compute_words(params[name]['paths'][0], 
                  os.path.join(params[name]['paths'][1], name + '/words'),
                  **params[name]['kwargs'])

