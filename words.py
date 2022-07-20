"""
Create Datasets of sequences of compound words - record parameters
"""
import sys

from dataset.make_words import compute_words

params = {
    'baseline_t1': {
        'paths': ['dataset/tokens/t1', 'saves'],
        'kwargs': {
            'bar_tokens': None,
            'eos_tokens': (True, True),
            'type_tokens': True,
            'in_metric_t_types': [],
            'in_note_t_types': [],
            'attr_metric_t_types': [],
            'attr_note_t_types': [],
            'out_metric_t_types': [],
            'out_note_t_types': []
        }
    }
}


if __name__ == '__main__':
    
    # get name from command line
    name = sys.argv[1]

    compute_words(*params[name]['paths'], **params[name]['kwargs'])

