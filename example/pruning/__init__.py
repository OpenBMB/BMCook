from collections import defaultdict
import types
import torch
import bmtrain as bmp
from bmtrain.block_layer import storage_type_cuda, round_up
from .prune_func import m4n2_2d_greedy, m4n2_2d_best
import os

def get_trivial_mask(p):
    return torch.ones_like(p)

def get_masks(ordered_parameters, func=get_trivial_mask, white_list=[]):
    ordered_masks = []
    for name, param in ordered_parameters:
        if any([white_name in name for white_name in white_list]) and len(param.shape) == 2:
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
            storaged_mask[storage_type].storage()[to_offset_st: to_offset_end].copy_(contiguous_param.storage()[offset_st: offset_end])
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
    # _white_list = ['project', 'out', 'w_'] # gpt
    _white_list = ['proj', 'fc_'] # gpt-j

    @classmethod
    def compute_mask(cls, model, func, checkpoint=None):
        '''
        Compute the pruning mask for each weight matrix and combine masks to match the parameters stored in the optimizer.
        
        :param model: Model to prune.
        :param func: Function for computing the pruning mask.
        :param checkpoint: Path to save/load the pruning mask.
        '''
        assert (cls._model is None), "BMPrune.compute_mask() can only be called once."
        cls._model = model

        _masks = []
        storaged_masks = None
        storaged_masks_ = {}

        if checkpoint is not None:
            if os.path.exists(checkpoint):
                storaged_masks = torch.load(checkpoint, map_location='cpu')
                assert len(storaged_masks) == len(model.dec_layers._modules)

        for i, dec_layer in model.dec_layers._modules.items():
            ordered_parameters = [(k, v) for k, v in dec_layer.state_dict().items()]
            if storaged_masks is not None:
                ordered_masks = storaged_masks[i]
                names = [k for k, _ in ordered_masks]
                assert len(set(names)) == len(names)
                # match name order
                d = {k: v for k, v in ordered_masks}
                new_names = [k for k, _ in ordered_parameters]
                ordered_masks = [(k, d[k]) for k in new_names]

                storaged_mask = mask_storage(ordered_masks, dec_layer._storage_params, dec_layer._storage_info)
            else:
                ordered_masks = get_masks(ordered_parameters, func, white_list=cls._white_list)
                storaged_mask = mask_storage(ordered_masks, dec_layer._storage_params, dec_layer._storage_info)
                
                storaged_masks_[i] = ordered_masks

            _masks.append((dec_layer._storage_params, storaged_mask))

        if storaged_masks is None and checkpoint is not None and bmp.global_var.config["rank"] == 0:
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

        cls._optimizer = optimizer
        cls._optimizer.step_old = optimizer.step

        def _step(opt_self, *args, **kwargs):
            # prune gradients before step method
            with torch.no_grad():
                for p, mask in cls._masks:
                    for k in p:
                        if p[k].grad is not None: #thx pjudd
                            p[k].grad.mul_(mask[k])
            # call original optimizer step method
            rval = opt_self.step_old(*args, **kwargs)
            # prune parameters after step method
            with torch.no_grad():
                for p, mask in cls._masks:
                    for k in p:
                        p[k].mul_(mask[k])
            return rval
        cls._optimizer.step = types.MethodType(_step, cls._optimizer)
