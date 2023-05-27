import torch
import math
from torch import Tensor

from typing import Optional

gamma, zeta, epsilon = -.1, 1.1, 1e-6
beta = 2./3.
zero_point = torch.tensor((0 - gamma) / (zeta - gamma))

def l0_norm_term(loga: Tensor):
    r"""calculate the l0 norm term. For details, see paper
    'Structured Pruning of Large Language Models' <https://arxiv.org/abs/1910.04732>"""
    x = loga - beta * (math.log(- gamma / zeta))
    loss = torch.sigmoid(x).sum()
    return loss

def from_sparsity(s: float):
    sig_input = -math.log((1 / s) - 1)
    loga = - sig_input + torch.log(zero_point / (1 - zero_point)) * beta
    return loga

def determinate_mask(loga: Tensor):
    r"""Drop the stochastic sampling in func:sample, used for evaluation when training sprune mask"""
    sig = torch.sigmoid(torch.log(zero_point / (1 - zero_point)) * beta - loga)
    s = 1 - sig
    out = s.clamp(min=epsilon, max=1-epsilon)
    return out

def sample(loga: Tensor):
    r"""Implements the gard concrete distribution. For details, see paper 
    'Learning Sparse Neural Networks through L_0 Regularization' <https://openreview.net/forum?id=H1Y8hhg0b>."""
    eps = loga.new_empty(loga.size()).uniform_(epsilon, 1-epsilon)
    s = torch.sigmoid((torch.log(eps / (1 - eps)) + loga) / beta)
    z = s.clamp(min=epsilon, max=1-epsilon) # remove from the computation graph of sp_module
    return z

def binarize(loga: torch.FloatTensor, hard_binarize: bool = False, target_s: Optional[float] = None):
    r"""According to the score mask, to get 0-1 mask for actual pruning."""
    dtype, device = torch.half, loga.device
    mask = determinate_mask(loga)

    if not hard_binarize:
        expected_num_nonzeros = mask.sum()
        total_num_nonzeros = loga.size(-1)
        num_zeros = round((total_num_nonzeros - expected_num_nonzeros).item())
    else:
        num_zeros = round(torch.tensor(mask.numel() * target_s).item())

    soft_mask = torch.ones_like(loga, dtype=dtype, device=device)
    if num_zeros > 0:
        if soft_mask.ndim == 0:
            soft_mask = torch.tensor(0, device=device)
        else:
            _, indices = torch.topk(mask, k=num_zeros, largest=False)  # return values, indices
            soft_mask[indices] = 0.  # set zero
    res = soft_mask
    
    return res
