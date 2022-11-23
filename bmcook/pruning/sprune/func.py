import torch
import math
from torch import Tensor

from typing import Optional

gamma, zeta, epsilon = -.1, 1.1, 1e-6
beta = 2./3.

def l0_norm_term(loga: Tensor):
    r"""calculate the l0 norm term. For details, see paper
    'Structured Pruning of Large Language Models' <https://arxiv.org/abs/1910.04732>"""
    x = loga - beta * (math.log(- gamma / zeta))
    loss = torch.sigmoid(x).sum()
    return loss

def from_sparsity(s: float):
    s = 1 - s
    sig = (s - gamma) / (zeta - gamma)
    loga = -math.log((1 / sig) - 1)
    return loga

def determinate_mask(loga: Tensor):
    sig = torch.sigmoid(loga)
    s = sig * (zeta - gamma) + gamma
    out = s.clamp(min=epsilon, max=1-epsilon)
    return out

def sample(loga: Tensor):
    r"""Implements the gard concrete distribution. For details, see paper 
    'Learning Sparse Neural Networks through L_0 Regularization' <https://openreview.net/forum?id=H1Y8hhg0b>."""
    eps = torch.FloatTensor(*loga.shape).uniform_(epsilon, 1-epsilon).to(loga.device)
    s = torch.sigmoid((torch.log(eps / (1 - eps)) + loga) / beta)
    s = s * (zeta - gamma) + gamma
    z = s.clamp(min=epsilon, max=1-epsilon)
    z_ = z.to(device='cuda', dtype=torch.half)  # remove from the computation graph of sp_module
    return z_

def binarize(loga: Tensor, hard_binarize: bool = False, target_s: Optional[float] = None):
    mask = determinate_mask(loga)
    if not hard_binarize:
        expected_num_nonzeros = torch.sum(mask, -1)
        total_num_nonzeros = loga.size(-1)
        expected_num_zeros = total_num_nonzeros - expected_num_nonzeros
    else:
        expected_num_zeros = torch.tensor(mask.numel() * target_s)
    if loga.size(-1) == 1:
        loga_ = loga.squeeze(-1)
        num_zeros = round(expected_num_zeros.item())
        soft_mask = torch.ones_like(loga_, dtype=torch.half)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga_.device)
            else:
                _, indices = torch.topk(loga_, k=num_zeros, largest=False)  # return values, indices
                soft_mask[indices] = 0.  # set zero
        res = soft_mask.to(device='cuda')  # remove from the computation graph of sp_module
    else:
        res = []
        for index in range(loga.size(0)):
            submask = mask[index]
            if not hard_binarize:
                expected_num_nonzeros = torch.sum(submask, -1)
                total_num_nonzeros = loga.size(-1)
                expected_num_zeros = total_num_nonzeros - expected_num_nonzeros
            else:
                expected_num_zeros = torch.tensor(submask.numel() * target_s)
            num_zeros = round(expected_num_zeros.item())
            cur_layer = loga[index]
            soft_mask = torch.ones_like(cur_layer, dtype=torch.half)
            if num_zeros > 0:
                if soft_mask.ndim == 0:
                    soft_mask = torch.tensor(0).to(cur_layer.device)
                else:
                    _, indices = torch.topk(mask[index], k=num_zeros, largest=False)  # 返回values, indices
                    soft_mask[indices] = 0.  # 置零
            res.append(soft_mask)
        res = torch.stack(res)
    return res


def binarize_mask_1d(loga: Tensor):
    expected_num_nonzeros = torch.sum(1 - cdf_concrete_dist(torch.tensor(0.), loga), -1)
    expected_num_zeros = loga.size(1) - expected_num_nonzeros
    
    num_zeros = round(expected_num_zeros)
    soft_mask = torch.ones_like(loga)
    if num_zeros > 0:
        if soft_mask.ndim == 0:
            soft_mask = torch.tensor(0).to(loga.device)
        else:
            _, indices = torch.topk(loga, k=num_zeros, largest=False)  # return values, indices
            soft_mask[indices] = 0.  # set zero
    soft_mask = soft_mask.to(device='cuda', dtype=torch.half)  # remove from the computation graph of sp_module
    return soft_mask