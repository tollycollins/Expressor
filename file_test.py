"""
tests
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from workspace.scripts import Controller


def test_lr(init_lr=3e-3,
            wu_factor=0.5,
            wu_ratio=0.1,
            min_lr=1e-8,
            restart_len=10,
            max_epochs=100, 
            restart_proportion=0.15):
    
    wu_len = wu_ratio * max_epochs
    lr_func = Controller.LR_Func(wu_factor, wu_len, min_lr, restart_len, max_epochs, 
                                 restart_proportion)
    
    lr = init_lr
    lrs = []
    epochs = np.arange(1, max_epochs + 1)
    
    # calculate learning rates
    for epoch in epochs:
        ratio = lr_func()
        lr = init_lr * ratio
        lrs.append(lr)

        print(f"Epoch {epoch}/{max_epochs}: lr: {lr} | ratio: {ratio}")
    
    # plot
    fig = plt.figure(dpi=100)
    plt.title('Learning Rate')
    plt.plot(epochs, lrs)
    plt.xlabel('epoch')
    plt.yscale('log')

    save_path = 'workspace/meta/lr.png'
    plt.savefig(save_path)

    plt.show()


if __name__ == '__main__':

    argv = sys.argv[1:]
    args = [arg for arg in argv if arg.find('=') < 0]
    kwargs = {kw[0]: float(kw[1]) for kw in [ar.split('=') for ar in argv if ar.find('=') > 0]}

    locals()[args[0]](**kwargs)
    
    