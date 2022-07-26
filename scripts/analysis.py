"""
Model performance analysis
"""
import os
import sys
import collections

import numpy as np
import matplotlib.pyplot as plt


def plot_training(path_log,
                  dpi=100):

    # load logfile
    monitor_vals = collections.defaultdict(list)
    with open(path_log, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                key, val, step, acc_time = line.split(' | ')
                monitor_vals[key].append((float(val), int(step), acc_time))
            except:
                continue

    # collect
    step_train = [item[1] for item in monitor_vals['epoch loss']]
    vals_train = [item[0] for item in monitor_vals['epoch loss']]

    step_valid = [item[1] for item in monitor_vals['validation loss']]
    vals_valid = [item[0] for item in monitor_vals['validation loss']]

    x_min = step_valid[np.argmin(vals_valid)]
    y_min = min(vals_valid)

    # plot
    fig = plt.figure(dpi=dpi)
    plt.title('training process')
    plt.plot(step_train, vals_train, label='train')
    plt.plot(step_valid, vals_valid, label='valid')
    plt.yscale('log')
    plt.plot([x_min], [y_min], 'ro')
    plt.legend(loc='upper right')
    plt.tight_layout()

    save_path = os.path.join(os.path.split(path_log)[0], 'learning_curve.png')
    plt.savefig(save_path)
    

if __name__ == '__main__':
    
    func = sys.argv[1]
    args = sys.argv[2:]

    print("here 1")

    if func == 'plot_training':

        print("here 2")
        
        path_log = os.path.join('saves', args[0], args[1], 'log.txt')
        plot_training(path_log)
