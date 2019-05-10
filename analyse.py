from __future__ import print_function, division

import argparse
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='logs/test')
    args = parser.parse_args()

    x = np.loadtxt(os.path.join(args.log_dir, 'positions.txt'))
    dim = x.shape[1]
    r = np.linalg.norm(x, axis=1)

    num, bins = np.histogram(r, density=False)
    center = (bins[:-1] + bins[1:]) / 2
    width = 0.7 * (bins[1] - bins[0])
    volume = bins[1:]**dim- bins[:-1]**dim
    plt.bar(center, num / volume, align='center', width=width)
    plt.savefig(os.path.join(args.log_dir, 'hist.png'))

    fig = plt.figure()
    if dim == 2:
        plt.scatter(x[:, 0], x[:, 1], marker='o', s=4)
    elif dim == 3:
        ax = Axes3D(fig)
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker='o', s=4)
    plt.savefig(os.path.join(args.log_dir, 'positions.png'))
