import types
import torch
from torch import Tensor
from torch.nn import Module
from typing import List, Dict, Union, Optional, NewType

import bmtrain as bmt

SPlugin = NewType('Plugin', Dict[str, Optional[Union[int, Tensor]]])

################################# get config from model #################################
def get_dim_ff(module: Module):
    if hasattr(module, 'dim_ff') and isinstance(module.dim_ff, int):
        ret = module.dim_ff
    else:
        ret = module.w_out.dim_in
    return ret

def get_dim_model(module: Module):
    return module.w_out.dim_out

################################# utils for forward-inside pruning setup #################################
def set_pruning_transformer(
    module: Module, 
    unit,
    is_bmtCBlock: bool = True 
    ):
    if is_bmtCBlock:
        module.forward_unprune = module._module.forward
    else:
        module.forward_unprune = module.forward
    def prune_forward(module_self, self_hidden_states, *args, **kwargs):
        mask = unit.mask.to(self_hidden_states.device)
        out = module_self.forward_unprune(self_hidden_states, *args, **kwargs)
        out = self_hidden_states + (out - self_hidden_states) * mask
        return out
    if is_bmtCBlock:
        module._module.forward = types.MethodType(prune_forward, module)
    else:
        module.forward = types.MethodType(prune_forward, module)

def set_pruning_att(
    module: Module, 
    index: int, 
    ATTENTION_MASK: List[SPlugin], 
    NUM_HEADS_MASK: List[SPlugin], 
    DIM_HEAD_MASK: List[SPlugin]
    ):
    module.index = index
    module.forward_unprune = module.forward
    def prune_forward(module_self, hidden_states, *args, **kwargs):
        mask = ATTENTION_MASK[module_self.index].mask.to(hidden_states.device)
        out = module_self.forward_unprune(hidden_states, *args, **kwargs)
        out = hidden_states + (out - hidden_states) * mask
        return out
    module.forward = types.MethodType(prune_forward, module)

    for s_name, s_module in module.named_modules():
        if 'project' in s_name:
            set_pruning_linear_attention(s_module, index, NUM_HEADS_MASK, DIM_HEAD_MASK, 'in')
        elif 'attention_out' in s_name:
            set_pruning_linear_attention(s_module, index, NUM_HEADS_MASK, DIM_HEAD_MASK, 'out')

def set_pruning_ffn(
    module: Module, 
    index: int, 
    ffn_m,
    DIM_FF_MASK: List[SPlugin]
    ):
    module.index = index
    module.forward_unprune = module.forward
    def prune_forward(module_self, hidden_states, *args, **kwargs):
        mask = ffn_m.mask
        out = module_self.forward_unprune(hidden_states, *args, **kwargs)
        out = hidden_states + (out - hidden_states) * mask
        return out
    module.forward = types.MethodType(prune_forward, module)

    for s_name, s_module in module.named_modules():
        if 'w_in.w' in s_name:
            set_pruning_linear_feedforward(s_module, index, DIM_FF_MASK, 'in')
        elif 'w_out' in s_name:
            set_pruning_linear_feedforward(s_module, index, DIM_FF_MASK, 'out')

def set_pruning_linear_attention(
    module: Module, 
    num_heads_unit,
    dim_head_unit,
    in_out: str,
    is_num_priority: bool = True
    ):
    module.forward_unprune = module.forward
    if in_out == 'in':
        def prune_forward(module_self, *args, **kwargs):
            num_heads, dim_head = num_heads_unit.dim, dim_head_unit.dim
            num_heads_mask, dim_head_mask = num_heads_unit.mask, dim_head_unit.mask
            out = module_self.forward_unprune(*args, **kwargs)  # (batch, len, num_heads*dim_head)

            old_size = out.size()
            out = out.view(old_size[0], old_size[1], num_heads, dim_head)
            if is_num_priority:
                out = out * num_heads_mask[:, None].to(out.device) * dim_head_mask.to(out.device)
            else:
                out = out * dim_head_mask.to(out.device) * num_heads_mask[:, None].to(out.device)
            out = out.view(old_size[0], old_size[1], num_heads * dim_head)
            return out

    elif in_out == 'out':
        def prune_forward(module_self, x, *args, **kwargs):
            num_heads, dim_head = num_heads_unit.dim, dim_head_unit.dim
            num_heads_mask, dim_head_mask = num_heads_unit.mask, dim_head_unit.mask

            old_size = x.size()  # (batch, len, num_heads * dim_head)
            x = x.view(old_size[0], old_size[1], num_heads, dim_head)
            if is_num_priority:
                x = x * num_heads_mask[:, None].to(x.device) * dim_head_mask.to(x.device)
            else:
                x = x * dim_head_mask.to(x.device) * num_heads_mask[:, None].to(x.device)
            x = x.view(old_size[0], old_size[1], num_heads * dim_head)

            out = module_self.forward_unprune(x, *args, **kwargs)  # (batch, len, dim_model)
            return out

    module.forward = types.MethodType(prune_forward, module)

def set_pruning_linear_feedforward(
    module: Module, 
    unit,
    in_out: str
    ):
    module.forward_unprune = module.forward
    if in_out == 'in':
        def prune_forward(module_self, *args, **kwargs):
            mask = unit.mask
            out = module_self.forward_unprune(*args, **kwargs)
            out = out * mask.to(out.device)
            return out
    elif in_out == 'out':
        def prune_forward(module_self, x, *args, **kwargs):
            mask = unit.mask
            x = x * mask.to(x.device)
            out = module_self.forward_unprune(x, *args, **kwargs)
            return out
    module.forward = types.MethodType(prune_forward, module)