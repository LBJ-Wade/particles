from __future__ import print_function, division

import argparse
import os

import torch
import numpy as np


def optimise(num_particles, dim, lam, num_iters=5000, lr=0.1, log_dir='', output_iter=100, device=None):

    device = torch.device('cuda' if device is None and torch.cuda.is_available() else 'cpu')

    # Initialise random particle positions
    x = torch.randn(num_particles, dim, requires_grad=True, device=device)

    opt = torch.optim.Adam([x], lr=lr)

    # Create log directory if it doesn't exist
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for i in range(num_iters):
        opt.zero_grad()
        V = lam / 6 * torch.sum(torch.norm(x, dim=1)**2)
        for x_row in x:
            d = torch.norm(x_row - x, dim=1, p=2)
            V += torch.sum(1 / d[d > 0]) / 2
        V.backward()
        opt.step()
        if i % output_iter == 0:
            print('Iter [%i] V [%5.4f]' % (i, V.detach().cpu().numpy()))
            if log_dir:
                np.savetxt(os.path.join(log_dir, 'positions.txt'), x.detach().cpu().numpy(), fmt='%1.4e')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=3, help="Dimensionality")
    parser.add_argument('--iters', type=int, default=2000, help="Number of iters")
    parser.add_argument("--particles", type=int, default=100, help="Number of particles")
    parser.add_argument('--lam', type=float, default=3, help="Lambda")
    parser.add_argument('--log_dir', type=str, default='logs/test')
    args = parser.parse_args()

    optimise(args.particles, args.dim, args.lam, num_iters=args.iters, log_dir=args.log_dir)
