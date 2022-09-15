from collections import defaultdict
import types
import torch
import bmtrain as bmt
from bmtrain.block_layer import storage_type_cuda, round_up
from .prune_func import m4n2_1d, m4n2_2d_greedy
import os
from bmcook.utils.config import ConfigParser
from .sprune import L0_Module_coarse, L0_Module_fine

def get_trivial_mask(p):
    return torch.ones_like(p)

def get_masks(ordered_parameters, func=get_trivial_mask, targets=[]):
    ordered_masks = []
    for name, param in ordered_parameters:
        if any([name == white_name for white_name in targets]) and len(param.shape) == 2:
            mask = func(param)
        else:
            mask = get_trivial_mask(param)
        ordered_masks.append((name, mask))
    return ordered_masks

def mask_storage(ordered_masks, storage_params, storage_info):
    storaged_mask = {}
    offsets = defaultdict(int)
    for k, v in storage_params.items():
        storaged_mask[k] = torch.ones_like(v) # not param

    for _, mask in ordered_masks:
        storage_type = storage_type_cuda(mask.storage_type())
        if storage_type == torch.cuda.HalfStorage:
            storage_type = 'float16_grad'
        param_shape = mask.size()
        param_st = offsets[storage_type]
        offsets[storage_type] += param_shape.numel()
        param_end = offsets[storage_type]
        offsets[storage_type] = round_up(offsets[storage_type], 512 // mask.element_size())
        storage_st = storage_info[storage_type]["begin"]
        storage_end = storage_info[storage_type]["end"]
        if not (param_st >= storage_end or param_end <= storage_st):
            # copy offset in parameter storage
            contiguous_param = mask.contiguous()
            offset_st = max(storage_st - param_st, 0)
            offset_end = min(storage_end - param_st, contiguous_param.numel())
            assert offset_st < offset_end

            # copy to offset in buffer storage
            to_offset_st = offset_st + param_st - storage_st
            to_offset_end = offset_end + param_st - storage_st

            # copy to buffer
            #storaged_mask[storage_type].storage()[to_offset_st: to_offset_end].copy_(contiguous_param.storage()[offset_st: offset_end])
            d_dtype = storaged_mask[storage_type].dtype
            d_device = storaged_mask[storage_type].device
            contiguous_param = contiguous_param.to(device=d_device)
            
            #if config['rank'] == 0:
            #    print(config['rank'], d_device, contiguous_param.device)
            assert d_device == contiguous_param.device, "The devices does not match, which is not allowed when duplicating storage."
            torch.tensor([], dtype=d_dtype, device=d_device).set_(storaged_mask[storage_type].storage(), to_offset_st, (to_offset_end - to_offset_st,))[:] = \
                        torch.tensor([], dtype=d_dtype, device=d_device).set_(contiguous_param.storage(), offset_st, (offset_end - offset_st,))[:]
            del contiguous_param
    return storaged_mask


class BMPrune:
    '''
    BMPrune prunes unimportant weights in PLMs.

    It consist of two steps: (1) Compute the pruning mask for each weight matrix. (2) Modify the optimizer to avoid the update of the pruned weights.
    '''

    _model = None
    _masks = None
    _optimizer = None
    _sprune_module = None

    @classmethod
    def compute_mask(cls, model, config):
        '''
        Compute the pruning mask for each weight matrix and combine masks to match the parameters stored in the optimizer.
        
        :param model: Model to prune.
        :param config: Configuration of the pruning.
        '''
        prune_config = config.get("pruning")
        if not prune_config["is_pruning"]:
            return

        assert (cls._model is None), "BMPrune.compute_mask() can only be called once."
        cls._model = model

        _masks = []
        storaged_masks = None
        storaged_masks_ = {}

        
        checkpoint = prune_config['pruning_mask_path'] if 'pruning_mask_path' in prune_config else None
        if prune_config['mask_method'] == 'm4n2_1d':
            func = m4n2_1d
        elif prune_config['mask_method'] == 'm4n2_2d':
            func = m4n2_2d_greedy
        elif prune_config['mask_method'] == 'coarse-grained':
            sprune_config = ConfigParser('cpm_live_example/config/l0_pruning.json').get('coarse-grained')
            if not sprune_config['train_mask']:
                mask = torch.load(sprune_config['coarse_mask'])
                return False, mask
            else:
                sprune_module = L0_Module_coarse(model, sprune_config)
                cls._optim_and_scheduler = sprune_module.create_sprune_optimizer()
                cls._sprune_module = sprune_module
                return True, None
        elif prune_config['mask_method'] == 'fine-grained':
            sprune_config = ConfigParser('cpm_live_example/config/l0_pruning.json').get('fine-grained')
            if not sprune_config['train_mask']:
                mask = torch.load(sprune_config['heads_mask'])
                return False, mask
            else:
                sprune_module = L0_Module_fine(sprune_config)
                cls._optim_and_scheduler = sprune_module.create_sprune_optimizer()
                cls._sprune_module = sprune_module
                return True, None
        else:
            raise ValueError("Unknown mask method: {}".format(prune_config['mask_method']))


        if checkpoint is not None:
            if os.path.exists(checkpoint):
                storaged_masks = torch.load(checkpoint, map_location='cpu')

        target_param_names = []
        for k in model.state_dict().keys():
            if any([pattern in k for pattern in prune_config['pruned_module']]):
                target_param_names.append(k)

        # for CheckpointBlock
        used_params = []
        for prefix, dec_layer in model.named_modules():
            if bmt.block_layer.CheckpointBlock == type(dec_layer):
                ordered_parameters = [(k, v) for k, v in dec_layer.state_dict().items()]
                filtered_targets = [x[len(prefix)+1:] for x in target_param_names if x.startswith(prefix)]

                if storaged_masks is not None:
                    ordered_masks = storaged_masks[prefix]
                    names = [k for k, _ in ordered_masks]
                    assert len(set(names)) == len(names)
                    # match name order
                    d = {k: v for k, v in ordered_masks}
                    new_names = [k for k, _ in ordered_parameters]
                    ordered_masks = [(k, d[k]) for k in new_names]

                    storaged_mask = mask_storage(ordered_masks, dec_layer._storage_params, dec_layer._storage_info)
                else:
                    ordered_masks = get_masks(ordered_parameters, func, targets=filtered_targets)
                    storaged_mask = mask_storage(ordered_masks, dec_layer._storage_params, dec_layer._storage_info)
                    
                    storaged_masks_[prefix] = ordered_masks
                
                used_params.extend([prefix+'.'+x for x in filtered_targets])
                _masks.append((dec_layer._storage_params, storaged_mask, True))    

        target_param_names = list(set(target_param_names) - set(used_params))

        for prefix, dec_layer in model.named_modules():

            if prefix+'.weight' in target_param_names:
                if storaged_masks is not None:
                    mask = storaged_masks[prefix]
                else:
                    mask = func(dec_layer.weight.data)
                    storaged_masks_[prefix] = mask
                
                _masks.append((dec_layer.weight, mask, False))


        if storaged_masks is None and checkpoint is not None and bmt.global_var.config["rank"] == 0:
            torch.save(storaged_masks_, checkpoint)

        cls._masks = _masks

    @classmethod
    def set_optim_for_pruning(cls, optimizer):
        '''
        Modify the step function of the optimizer to avoid the update of the pruned weights, i.e., setting corresponding gradients and parameters to zeros.

        :param optimizer: Optimizer to modify.
        :return: Modified optimizer.
        '''

        assert (cls._optimizer is None), "BMPrune.set_optim_for_pruning() can only be called once."

        if cls._model is None:
            return

        cls._optimizer = optimizer
        cls._optimizer.step_old = optimizer.step

        def _step(opt_self, *args, **kwargs):
            # prune gradients before step method
            with torch.no_grad():
                for p, mask, flag in cls._masks:
                    if flag:
                        for k in p:
                            if p[k].grad is not None:
                                p[k].grad.mul_(mask[k])
                    else:
                        if p.grad is not None:
                            p.grad.mul_(mask)
            # call original optimizer step method
            rval = opt_self.step_old(*args, **kwargs)
            # prune parameters after step method
            with torch.no_grad():
                for p, mask, flag in cls._masks:
                    if flag:
                        for k in p:
                            p[k].mul_(mask[k])
                    else:
                        tmp_mask = torch.tensor(mask, dtype=p.dtype, device=p.device)
                        p.mul_(tmp_mask)
            return rval
        cls._optimizer.step = types.MethodType(_step, cls._optimizer)
