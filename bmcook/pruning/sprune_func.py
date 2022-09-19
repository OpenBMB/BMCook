import torch

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
beta = 2./3.

"""Implements the CDF of the 'stretched' concrete distribution"""
def cdf_concrete_dist(eps, loga):
    xn = (eps - limit_a) / (limit_b - limit_a)  # 0.083333
    s = torch.sigmoid((torch.log(xn / (1 - xn)) * beta - loga))
    z = s.clamp(min=epsilon, max=1 - epsilon)
    return z


def sample_mask(loga):
    eps = torch.FloatTensor(*loga.shape).uniform_(epsilon, 1-epsilon).to(loga.device)
    s = torch.sigmoid((torch.log(eps / (1 - eps)) + loga) / beta)
    s = s * (limit_b - limit_a) + limit_a
    z = s.clamp(min=0., max=1.)
    return z


def binarize_mask(loga, size):
    expected_num_nonzeros = torch.sum(1 - cdf_concrete_dist(torch.tensor(0.), loga))
    expected_num_zeros = size - expected_num_nonzeros.item()
    num_zeros = round(expected_num_zeros)
    soft_mask = torch.ones_like(loga)
    if num_zeros > 0:
        if soft_mask.ndim == 0:
            soft_mask = torch.tensor(0).to(loga.device)
        else:
            _, indices = torch.topk(loga, k=num_zeros, largest=False)  # 返回values, indices
            soft_mask[indices] = 0.  # 置零
    return soft_mask