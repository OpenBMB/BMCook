import torch
from torch import Tensor
from typing import Dict
from collections import OrderedDict

from .unit import SPruneUnit

def unit_list_decorator(func):
    """
    unit_list_decorator provides a mapping between index and module name, to support a module name based access to sprune unit.
    """
    def decorator(module_self, *args):
        __o = args[-1]
        if not isinstance(__o, SPruneUnit):
            raise TypeError(f"SPrunePlugin should maintain lists of SPruneUnit, but got {type(__o)}")
        module_self._item_mapping[__o.name] = __o
        return func(module_self, *args) 
    return decorator

class UnitList(list):
    def __init__(self, *args, **kwargs):
        super(UnitList, self).__init__(*args, **kwargs)
        self._item_mapping: Dict = OrderedDict()  # {name: unit}

    @unit_list_decorator
    def append(self, __object):
        return super().append(__object)

    @unit_list_decorator
    def insert(self, __index, __object):
        return super().insert(__index, __object)

    @unit_list_decorator
    def __setitem__(self, __i, __o):
        return super().__setitem__(__i, __o)
    
    def __delitem__(self, __i):
        __o = self[__i]
        del self._item_mapping[__o.name]
        return super().__delitem__(__i)

    def from_name(self, name):
        if name in self._item_mapping:
            ret = self._item_mapping[name]
        else:
            ret = None
        return ret
    
    def name2index(self, name):
        return self.index(self.from_name(name))

    def get_mask(self):
        if len(self) is None:
            ret = None
        else:
            ret = torch.stack([k.mask for k in self])
        return ret

    def load_mask(self, masks: Tensor):
        assert len(self) == masks.size(0)
        for i, _ in enumerate(self):
            self[i].mask = masks[i]

    def get_max_dim(self):
        max_dim = max([item.dim for item in self])
        return max_dim

    def binarize_mask(self, hard_binarize: bool = False, target_s: float = None):
        r"""According to unit's density, get corresponding 0-1 mask."""
        mask = torch.stack([unit.mask for unit in self]).float()  # 2-d
        soft_mask = torch.ones_like(mask)

        if mask.size(1) == 1:  # ModuleUnit pruning
            expected_num_nonzeros = mask.sum()
            total_num_nonzeros = mask.size(0)
            num_zeros = round((total_num_nonzeros - expected_num_nonzeros).item()) if hard_binarize else \
                            round(torch.tensor(mask.numel() * target_s).item())

            if num_zeros > 0:
                _, indices = torch.topk(mask.squeeze(-1), k=num_zeros, largest=False)  # return values, indices
                soft_mask[indices, :] = 0.  # set zero
        else:  # ParamUnit pruning
            for i, mask_1d in enumerate(mask):
                expected_num_nonzeros = mask_1d.sum()
                total_num_nonzeros = mask_1d.size(0)
                num_zeros = round((total_num_nonzeros - expected_num_nonzeros).item()) if hard_binarize else \
                                round(torch.tensor(mask.numel() * target_s).item())
                
                if num_zeros > 0:
                    _, indices = torch.topk(mask_1d, k=num_zeros, largest=False)  # return values, indices
                    soft_mask[i, indices] = 0.  # set zero
        
        for i, unit_mask in enumerate(soft_mask):
            self[i].mask = unit_mask.half()