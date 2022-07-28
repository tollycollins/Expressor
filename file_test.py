"""
tests
"""
import argparse

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
    lr_func = Controller.LR_Func(init_lr, wu_factor, wu_len, min_lr, restart_len, max_epochs)
    
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
    plt.show()

    save_path = 'workspace/meta/lr.png'
    plt.savefig(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("func")
    parser.add_argument("--init_lr")
    parser.add_argument("--wu_factor")
    parser.add_argument("--wu_ratio")
    parser.add_argument("--min_lr")
    parser.add_argument("--restart_len")
    parser.add_argument("--max_epochs")
    parser.add_argument("--restart_proportion")
    
    args_dict = vars(parser.parse_args())

    try:
        kwargs = args_dict['keyword_args']
    except KeyError:
        kwargs = {}

    func = args_dict['func']
    locals()[args_dict['func']](**kwargs)
    
    