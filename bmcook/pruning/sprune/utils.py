import torch
import fnmatch
import bmtrain.nccl as nccl
from torch.nn import Module

import bmtrain as bmt

################################# get config from model #################################
def get_dim_ff(module: Module):
    if hasattr(module, 'dim_ff') and isinstance(module.dim_ff, int):
        ret = module.dim_ff
    else:
        ret = module.w_out.dim_in
    return ret

def get_dim_model(module: Module):
    return module.w_out.dim_out

def inspect_checkpoint_block(model : bmt.CheckpointBlock, param_name : str, prefix : str = ''):
    # fast check
    pass_fast_check = False
    for param in model._param_info:
        abs_name = prefix + param["name"]
        if fnmatch.fnmatch(abs_name, param_name):
            pass_fast_check = True
            break
    if not pass_fast_check:
        return []

    _param_buffer = {}
    _grad_buffer = {}
    for kw, val in model._storage_info.items():
        storage_type = model._storage_params[kw].storage_type()

        _param_buffer[kw] = storage_type(val["partition_size"] * bmt.global_var.config['world_size'])
        if model._storage_params[kw].grad is not None:
            _grad_buffer[kw] = storage_type(val["partition_size"] * bmt.global_var.config['world_size'])
    
    nccl.groupStart()
    for kw, val in model._storage_info.items():
        nccl.allGather(
            model._storage_params[kw].storage(),
            _param_buffer[kw],
            bmt.global_var.config["comm"]
        )
        if model._storage_params[kw].grad is not None:
            nccl.allGather(
                model._storage_params[kw].grad.storage(),
                _grad_buffer[kw],
                bmt.global_var.config["comm"]
            )

    nccl.groupEnd()
    ret = []
    for param in model._param_info:
        abs_name = prefix + param["name"]
        if fnmatch.fnmatch(abs_name, param_name):
            kw_name = param["kw_name"]
            dtype = _param_buffer[kw_name].dtype
            device = _param_buffer[kw_name].device
            offset = param["offset"]
            shape = param["shape"]
            p = torch.tensor([], dtype=dtype, device=device).set_(_param_buffer[kw_name], offset, shape)
            if kw_name in _grad_buffer:
                g = torch.tensor([], dtype=dtype, device=device).set_(_grad_buffer[kw_name], offset, shape)
                ret.append({
                    "name": abs_name,
                    "param": p
                })
            else:
                ret.append({
                    "name": abs_name,
                    "param": p
                })
    return ret
