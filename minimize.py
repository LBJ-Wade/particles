from __future__ import print_function, division

import argparse
import os

import torch
import numpy as np


def pairwise_potential(x, masses):
    mass_matrix = masses[:, None] * masses
    res = torch.norm(1/ (x[:, None] - x), dim=2, p=2)
    return res


def optimise(masses, dim, lam, num_iters=5000, lr=0.1, log_dir='', output_iter=100, device=None):

    device = torch.device('cuda' if device is None and torch.cuda.is_available() else 'cpu')

    masses = torch.from_numpy(masses).float().to(device)
    num_particles = masses.size()[0]

    # Initialise random particle positions
    x = torch.randn(num_particles, dim, requires_grad=True, device=device)

    # Indices of upper triangular distance matrix
    idx = torch.triu(torch.ones(num_particles, num_particles), diagonal=1) == 1

    opt = torch.optim.Adam([x], lr=lr)

    # Create log directory if it doesn't exist
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_dir:
        f = open(os.path.join(log_dir, 'log.txt'), 'w')

    for i in range(num_iters):
        opt.zero_grad()
        dist = pairwise_potential(x, masses)[idx]
        V = torch.sum(1 / dist) + lam / 6 * torch.sum(masses * torch.norm(x, dim=1) ** 2)
        V.backward()
        opt.step()
        if i % output_iter == 0 and log_dir:
                f.write('%i %5.4f \n' % (i, V.detach().cpu().numpy()))
                f.flush()
                np.savetxt(os.path.join(log_dir, 'positions.txt'), x.detach().cpu().numpy(), fmt='%1.4e')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=3, help="Dimensionality")
    parser.add_argument('--iters', type=int, default=2000, help="Number of iters")
    parser.add_argument("--particles", type=int, default=100, help="Number of particles")
    parser.add_argument('--lam', type=float, default=3, help="Lambda")
    parser.add_argument('--log_dir', type=str, default='logs/test')
    parser.add_argument('--masses', type=str)
    args = parser.parse_args()

    if args.masses:
        masses = np.loadtxt(args.masses)
    else:
        masses = np.ones(args.particles)

    optimise(masses, args.dim, args.lam, num_iters=args.iters, log_dir=args.log_dir)
