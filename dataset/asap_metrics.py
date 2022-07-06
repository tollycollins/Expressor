import sys
import os

import utils


def time_signature(print_names=[], show_composer=False, verbose=True):
    metadata = utils.get_metadata()
    time_sigs = []
    # Get list of time signatures, beats per bar, frequencies in dataset and 
    # number of tracks with this time signature
    for name, track in metadata.items():
        sigs = track['perf_time_signatures'].values()
        unique_sigs = set()
        for sig in sigs:
            new_sig = True
            for val in time_sigs:
                if sig[0] == val[0]:
                    val[2] += 1
                    new_sig = False
                    if tuple(sig) not in unique_sigs:
                        val[3] += 1
                        if sig[0] in print_names:
                            if not show_composer:
                                name = os.path.split(name)[1]
                            name = os.path.splitext(name)[0]
                            val[4].append(name)
                    break
            if new_sig:
                data = [*sig, 1, 1]
                if sig[0] in print_names:
                    if not show_composer:
                        name = os.path.split(name)[1]
                    name = os.path.splitext(name)[0]
                    data.append([name])
                time_sigs.append(data)
            unique_sigs.add(tuple(sig))
    
    time_sigs = sorted(time_sigs, reverse=True, key=lambda i: i[3])
    if verbose:
        print("Time signatures in dataset (sig, BPB, frequency, track_appearances):")
        print(*time_sigs, sep='\n')

    return time_sigs


def pitches():
    pass


if __name__ == '__main__':
    globals()[sys.argv[1]](*sys.argv[2:])

