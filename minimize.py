from __future__ import print_function, division

import argparse
import os

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def optimise(num_particles, dim, lam, num_iters=5000):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(num_particles, dim, requires_grad=True).to(device)

    opt = torch.optim.Adam([x], lr=0.1)

    for i in range(num_iters):
        opt.zero_grad()
        V = torch.sum(1 / torch.nn.functional.pdist(x)) + lam / 6 * torch.sum(torch.norm(x, dim=1))
        V.backward()
        opt.step()
        if i % 100 == 0:
            print('Iter [%i] V [%5.4f]' % (i, V.detach().numpy()))

    return x.detach().numpy()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dim', type=int, default=3, help="Dimensionality")
    parser.add_argument('--iters', type=int, default=2000, help="Number of iters")
    parser.add_argument("--particles", type=int, default=100, help="Number of particles")
    parser.add_argument('--lam', type=float, default=3, help="Lambda")
    parser.add_argument('--log_dir', type=str, default='logs/test')
    parser.add_argument('-plot', action='store_true')

    args = parser.parse_args()

    x = optimise(args.particles, args.dim, args.lam, num_iters=args.iters)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    np.savetxt(os.path.join(args.log_dir, 'positions.txt'), x)

    if args.plot:

        r = np.linalg.norm(x, axis=1)

        num, bins = np.histogram(r, density=False)
        center = (bins[:-1] + bins[1:]) / 2
        width = 0.7 * (bins[1] - bins[0])
        volume = bins[1:]**args.dim - bins[:-1]**args.dim
        plt.bar(center, num / volume, align='center', width=width)
        plt.show()

        fig = plt.figure()
        if x.shape[1] == 2:
            plt.scatter(x[:,0], x[:,1], marker='o', s=4)
        elif x.shape[1] == 3:
            ax = Axes3D(fig)
            ax.scatter(x[:,0], x[:,1], x[:,2], marker='o', s=4)
        plt.show()
