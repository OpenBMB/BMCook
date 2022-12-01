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
                    'mask_modules': [],
                    'num_heads': [],
                    'dim_head': [],
                    'dim_ff': [],
                    }

    new_state_dict = state_dict.copy()
    
    for transformer_mask in BMPrune.sprune_engine.plugin:
        if transformer_mask.training is True and transformer_mask.mask == 0:
            for k in state_dict:
                if k.startswith(transformer_mask.name):
                    del new_state_dict[k]
        att_mask = transformer_mask.att
        ffn_mask = transformer_mask.ffn
        prune_att = False
        prune_ffn = False
        if att_mask.training is True and att_mask.mask == 0:
            prune_att = True
            for k in state_dict:
                if k.startswith(att_mask.name) and k in new_state_dict:
                    del new_state_dict[k]
        if ffn_mask.training is True and ffn_mask.mask == 0:
            prune_ffn = True
            for k in state_dict:
                if k.startswith(ffn_mask.name) and k in new_state_dict:
                    del new_state_dict[k]
        
        num_heads_mask = att_mask.num_heads
        dim_head_mask = att_mask.dim_head
        prune_num_heads = num_heads_mask.dim
        prune_dim_head = dim_head_mask.dim

        if num_heads_mask.training is True or dim_head_mask.training is True:
            num_heads_indices = torch.nonzero(num_heads_mask.mask == 1., as_tuple=True)[0]
            dim_head_indices = torch.nonzero(dim_head_mask.mask == 1., as_tuple=True)[0]
            new_num_heads_dim = num_heads_indices.size(0)
            new_dim_head_dim = dim_head_indices.size(0)
            for k in state_dict:
                if k.startswith(num_heads_mask.name) and k in new_state_dict:  # num_heads_mask.name == dim_head_mask.name
                    old_weight = new_state_dict[k]
                    old_num_heads = num_heads_mask.dim
                    old_dim_head = dim_head_mask.dim
                    old_weight = old_weight.view(old_num_heads, old_dim_head, -1) if '.attention_out.weight' not in k else old_weight.permute(1, 0).contiguous().view(old_num_heads, old_dim_head, -1)
                    new_weight = old_weight[num_heads_indices, :, :][:, dim_head_indices, :]
                    new_weight = new_weight.view(new_num_heads_dim * new_dim_head_dim, -1) if '.attention_out.weight' not in k else new_weight.view(new_num_heads_dim*new_dim_head_dim, -1).permute(1, 0).contiguous()
                    new_state_dict[k] = new_weight
            prune_num_heads = new_num_heads_dim
            prune_dim_head = new_dim_head_dim

        dim_ff_mask = ffn_mask.dim_ff
        prune_dim_ff = dim_ff_mask.dim

        if dim_ff_mask.training is True:
            dim_ff_indices = torch.nonzero(dim_ff_mask.mask == 1., as_tuple=True)[0]
            new_dim_ff_dim = dim_ff_indices.size(0)
            old_dim_ff = dim_ff_mask.dim
            for k in state_dict:
                if k.startswith(dim_ff_mask.name) and k in new_state_dict:
                    old_weight = new_state_dict[k]
                    old_weight = old_weight.view(old_dim_ff, -1) if 'w_out.weight' not in k else old_weight.permute(1, 0).contiguous().view(old_dim_ff, -1)
                    new_weight = old_weight[dim_ff_indices, :]
                    new_weight = new_weight.view(new_dim_ff_dim, -1) if 'w_out.weight' not in k else new_weight.view(new_dim_ff_dim, -1).permute(1, 0).contiguous()
                    new_state_dict[k] = new_weight
            prune_dim_ff = new_dim_ff_dim
        
        model_config['mask_modules'].append(tuple([prune_att, prune_ffn]))
        model_config['num_heads'].append(prune_num_heads)
        model_config['dim_head'].append(prune_dim_head)
        model_config['dim_ff'].append(prune_dim_ff)

    if bmt.global_var.config["rank"] == 0:
        torch.save(new_state_dict, file_name)
    
        with open(file_name + '.cnofig.json', 'w') as f:
            json.dump(model_config, f)

def save_masks(file_name):
    if BMPrune.sprune_engine is not None:
        BMPrune.sprune_engine.save(file_name)

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

                new_state_dict[key] = quant_value.squeeze(0).to('cpu')
    
        torch.save(new_state_dict, file_name)


def save(model, file_name: str, mode: str):
    if mode == 'quant':
        _save_quantized(model, file_name)
    elif mode == 'pruning':
        _save_spruned(model, file_name)