from __future__ import print_function, division

import argparse
import os

import torch
import numpy as np


def optimise(num_particles, dim, lam, num_iters=5000, lr=0.1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(num_particles, dim, requires_grad=True, device=device)

    opt = torch.optim.Adam([x], lr=lr)

    for i in range(num_iters):
        opt.zero_grad()
        #V = torch.sum(1 / torch.nn.functional.pdist(x)) + lam / 6 * torch.sum(torch.norm(x, dim=1))
        V = torch.sum(torch.norm(x[:, None] - x, dim=2, p=2)) / 2 + lam / 6 * torch.sum(torch.norm(x, dim=1))
        V.backward()
        opt.step()
        if i % 100 == 0:
            print('Iter [%i] V [%5.4f]' % (i, V.detach().cpu().numpy()))

    return x.detach().cpu().numpy()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=3, help="Dimensionality")
    parser.add_argument('--iters', type=int, default=2000, help="Number of iters")
    parser.add_argument("--particles", type=int, default=100, help="Number of particles")
    parser.add_argument('--lam', type=float, default=3, help="Lambda")
    parser.add_argument('--log_dir', type=str, default='logs/test')
    args = parser.parse_args()

    x = optimise(args.particles, args.dim, args.lam, num_iters=args.iters)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    np.savetxt(os.path.join(args.log_dir, 'positions.txt'), x)
