import torch
import json
import bmtrain as bmt
import cpm_kernels.kernels as ck
from collections import OrderedDict

from .pruning import BMPrune

def _save_to_state_dict(model : torch.nn.Module, destination, prefix):
    if isinstance(model, bmt.CheckpointBlock):
        if bmt.global_var.config['rank'] != 0:
            destination = OrderedDict() # creates an temporary ordered dict
            destination._metadata = OrderedDict()
        model.state_dict(destination, prefix, False)
    else:
        if bmt.global_var.config['rank'] != 0:
            destination = OrderedDict() # creates an temporary ordered dict
            destination._metadata = OrderedDict()
        model._save_to_state_dict(destination, prefix, False)

def _save_to_rank0(model : torch.nn.Module, destination=None, prefix=''):
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=model._version)
    _save_to_state_dict(model, destination, prefix)
    for name, module in model._modules.items():
        if module is not None:
            _save_to_rank0(module, destination, prefix + name + '.')
    for hook in model._state_dict_hooks.values():
        hook_result = hook(model, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination

def _save_spruned(model, file_name):
    # get state_dict of model
    torch.cuda.synchronize()
    state_dict = _save_to_rank0(model)

    model_config = {
                    'att': {},
                    'ffn': {},
                    'num_heads': [],
                    'dim_head': [],
                    'dim_ff': [],
                    }

    new_state_dict = state_dict.copy()
    
    for name, list in BMPrune.sprune_engine.plugin.prunable_name_lists():
        for i, unit in enumerate(list):
            if unit.mask == 0:
                model_config[name][i] = True
                for k in state_dict:
                    if k.startswith(unit.name):
                        del new_state_dict[k]
            elif "params" in unit.list_in:
                if "att" in unit.list_in:
                    model_config['num_heads'].append(unit.num_heads.mask.sum().item())
                    model_config['dim_head'].append(unit.dim_head.mask.sum().item())
                else:
                    model_config['dim_ff'].append(unit.dim_ff.mask.sum().item())
                for k in state_dict:
                    if k.startswith(unit.name):
                        new_state_dict[k] = unit.prune_params(k, state_dict[k])
    
    if bmt.global_var.config["rank"] == 0:
        torch.save(new_state_dict, file_name)
    
        with open(file_name + '.cnofig.json', 'w') as f:
            json.dump(model_config, f)

def save_masks(file_name):
    if BMPrune.sprune_engine is not None:
        BMPrune.sprune_engine.save_masks(file_name)

def load_masks(file_name):
    if BMPrune.sprune_engine is not None:
        BMPrune.sprune_engine.plugin.load_masks(file_name)

def _save_quantized(model, file_name):
    torch.cuda.synchronize()
    state_dict = _save_to_rank0(model)

    if bmt.global_var.config['rank'] == 0:
        new_state_dict = state_dict.copy()

        for name, module in model.named_modules():
            if hasattr(module, "quant") and module.quant is True:
                key = name+".weight"
                value = state_dict[key].unsqueeze(0).cuda()

                assert value.is_cuda and value.is_contiguous() and value.dtype == torch.half

                scale_value = torch.empty((value.size(0), value.size(1)), dtype=torch.half, device=value.device)
                ck.gemm_calc_scale(
                    value.size(0), value.size(1), value.size(2),
                    value.data_ptr(), scale_value.data_ptr(), torch.cuda.current_stream().cuda_stream
                )

                quant_value = torch.empty(value.size(), dtype=torch.int8, device=value.device)
                ck.gemm_round(
                    value.size(0), value.size(1), value.size(2),
                    value.data_ptr(), scale_value.data_ptr(), quant_value.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )

                new_state_dict[key+"_quant"] = quant_value.squeeze(0).to('cpu')
                new_state_dict[key+"_scale"] = scale_value.squeeze(0).to("cpu")
                del new_state_dict[key]
    
        torch.save(new_state_dict, file_name)


def save(model, file_name: str, mode: str):
    if mode == 'quant':
        _save_quantized(model, file_name)
    elif mode == 'prune':
        _save_spruned(model, file_name)
