import os
import torch
import bmtrain as bmt
from model_center.layer import TransformerBlock
from model_center.model import BaseModel

from typing import Dict, List, Optional
from collections import OrderedDict

from .utils import *

GRAIN_MAPPING = {
    'transformer':  0,
    'att':          1,
    'ffn':          1,
    'num_heads':    2,
    'dim_head':     2,
    'dim_ff':       2,
}

class SPruneUnit:
    def __init__(
        self,
        name, 
        param, 
        dim: int = 1, 
        num: int = 1, 
        is_leaf: bool = False,
        is_same: bool = False
        ):
        self._subunits: Optional[SPruneUnit] = OrderedDict()
        self.name = name
        self.param = param
        self.num = num
        self.dim = dim
        self.is_leaf = is_leaf
        self.is_same = is_same
        self.residual = param * dim
        self.is_dropped = False
        if dim == 1:
            self.mask = torch.tensor(1., dtype=torch.half, device="cuda")
            self.density = torch.tensor(1., dtype=torch.float, device="cuda")
        else:
            self.mask = torch.ones(dim, dtype=torch.half, device="cuda")
            self.density = torch.ones(dim, dtype=torch.float, device="cuda")

    def __setattr__(self, key, value):
        if isinstance(value, SPruneUnit):
            _subunits = self.__dict__.get('_subunits')
            _subunits[key] = value
            if not value.is_same:
                self.residual -= (value.param * value.dim * value.num)
        self.__dict__[key] = value

    def get_param_exp(self):
        if self.is_dropped:
            return 0
        if self.is_leaf:
            res = torch.sum(self.density * self.param) * self.num
        else:
            res = self.density * self.residual
            for _, unit in self._subunits.items():
                res += self.density * unit.get_param_exp()
        return res
    
    def get_param_all(self):
        if self.is_dropped:
            return 0
        if self.is_leaf:
            res = torch.tensor(self.param) * self.num
        else:
            res = self.residual
            for _, unit in self._subunits.items():
                res += unit.get_param_all()
        return res

def unit_list_decorator(func):
    def decorator(module_self, *args):
        __o = args[-1]
        if not isinstance(__o, SPruneUnit):
            raise TypeError(f"SPrunePlugin should maintain lists of SPruneUnit, but got {type(__o)}")
        module_self._item_mapping[__o.name] = __o
        return func(module_self, *args) 
    return decorator

class UnitList(list):
    def __init__(self, name, *args, **kwargs):
        super(UnitList, self).__init__(*args, **kwargs)
        self._item_mapping: Dict = OrderedDict()
        self.is_leaf = True
        assert name in ["transformer", "att", "ffn", "num_heads", "dim_head", "dim_ff"]
        self.grain = GRAIN_MAPPING[name]

    @unit_list_decorator
    def append(self, __object):
        return super().append(__object)

    @unit_list_decorator
    def insert(self, __index, __object):
        return super().insert(__index, __object)

    @unit_list_decorator
    def __setitem__(self, __i, __o):
        return super().__setitem__(__i, __o)

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

    def set_leaf(self):
        for i, _ in enumerate(self):
            self[i].is_leaf = True
    
    def set_drop(self):
        for i, _ in enumerate(self):
            self[i].is_dropped = True

    def get_max_dim(self):
        dim_list = [item.dim for item in self]
        return max(dim_list)

    def iter_names(self):
        for n in self._item_mapping:
            yield n


class SPrunePlugin:
    r"""
    SPrunePlugin is a base class for structure prune in BMCook.
    All the modules supported by SprunePlugin includes: TransformerBlock layer, Attention layer, Feedforward layer, num_heads, dim_head, dim_ff.
    """
    def __init__(self, model: BaseModel, saved_path: Optional[str] = None) -> None:
        '''
        analyze the structure prune methed.
        '''
        self.training_masks: Dict[str, UnitList] = OrderedDict()
        self.base_grain: Optional[int] = None

        self.transformer = UnitList('transformer')
        self.att = UnitList('att')
        self.ffn = UnitList('ffn')
        self.num_heads = UnitList('num_heads')
        self.dim_head = UnitList('dim_head')
        self.dim_ff = UnitList('dim_ff')
        
        # model analysis
        num_encoder_layers, num_decoder_layers = 0, 0
        prunable_all_params, self_att_num = 0, 0
        cross_att_buffer, cross_num_heads_buffer, cross_dim_head_buffer, names_buffer = [], [], [], []
        
        for name, module in model.named_modules():
            if type(module) in (bmt.block_layer.CheckpointBlock, TransformerBlock):
                block_type = name.split('.')[0]
                if block_type == 'encoder':
                    num_encoder_layers += 1
                if block_type == 'decoder':
                    num_decoder_layers += 1

                ordered_parameters = [(k, v) for k, v in module.state_dict().items()]
                self_att_param, cross_att_param, ffn_param = 0, 0, 0
                dim_ff_num = 0
                for (k, v) in ordered_parameters:
                    if 'self_att' in k:
                        self_att_param += v.numel()
                    elif 'cross_att' in k:
                        cross_att_param += v.numel()
                    elif 'ffn' in k:
                        if 'w_in' in k or 'w_out' in k:
                            dim_ff_num += 1
                        ffn_param += v.numel()
                
                # model ststistics
                transformer_layer_param = self_att_param + cross_att_param + ffn_param
                prunable_all_params += transformer_layer_param

                if transformer_layer_param > 0:
                    cur_transformer = SPruneUnit(name, transformer_layer_param)
                    self.transformer.append(cur_transformer)
                    # set_pruning_transformer(module, len(self.transformer)-1, self.transformer, isinstance(module, bmt.CheckpointBlock))

                    if self_att_param > 0:
                        dim_head = module.self_att.self_attention.dim_head
                        num_heads = module.self_att.self_attention.num_heads
                        dim_model = module.self_att.self_attention.dim_model

                        cur_att = SPruneUnit(name+'.self_att', self_att_param)
                        cur_num_heads = SPruneUnit(name+'.self_att.self_attention', dim_head * dim_model, num_heads, num=4)
                        cur_dim_head = SPruneUnit(name+'.self_att.self_attention', num_heads * dim_model, dim_head, num=4, is_same=True)

                        cur_transformer.att = cur_att
                        cur_att.num_heads = cur_num_heads
                        cur_att.dim_head = cur_dim_head

                        self.att.append(cur_att)
                        self.num_heads.append(cur_num_heads)
                        self.dim_head.append(cur_dim_head)
                        # set_pruning_att(module.self_att, len(self.att)-1, self.att, self.num_heads, self.dim_head)
                        self_att_num += 1
                                           
                    if cross_att_param > 0:
                        cross_dim_head = module.cross_att.self_attention.dim_head
                        cross_num_heads = module.cross_att.self_attention.num_heads
                        dim_model = module.cross_att.self_attention.dim_model
                        names_buffer.append(name)

                        cur_cross_att = SPruneUnit(name+'.cross_att', cross_att_param)
                        cur_cross_num_heads = SPruneUnit(name+'.cross_att.self_attention', cross_dim_head * dim_model, cross_num_heads, num=4)
                        cur_cross_dim_head = SPruneUnit(name+'.cross_att.self_attention', cross_num_heads * dim_model, cross_dim_head, num=4, is_same=True)

                        cur_transformer.cross_att = cur_cross_att
                        cur_cross_att.cross_num_heads = cur_cross_num_heads
                        cur_cross_att.cross_dim_head = cur_cross_dim_head

                        cross_att_buffer.append(cur_transformer)
                        cross_num_heads_buffer.append(cur_cross_num_heads)
                        cross_dim_head_buffer.append(cur_cross_dim_head)

                    if ffn_param > 0:
                        dim_model = get_dim_model(module.ffn.ffn)
                        dim_ff = get_dim_ff(module.ffn.ffn)

                        cur_ffn = SPruneUnit(name+'.ffn', ffn_param)
                        cur_dim_ff = SPruneUnit(name+'.ffn.ffn', dim_model, dim_ff, num=dim_ff_num)

                        cur_transformer.ffn = cur_ffn
                        cur_ffn.dim_ff = cur_dim_ff

                        self.ffn.append(cur_ffn)
                        self.dim_ff.append(cur_dim_ff)
                        # set_pruning_ffn(module.ffn, len(self.ffn)-1, self.ffn, self.dim_ff)

        # append cross_att to att and set pruning
        for (module_name, cross_att, cross_num_heads, cross_dim_head) in \
            zip(names_buffer, cross_att_buffer, cross_num_heads_buffer, cross_dim_head_buffer):

            self.att.append(cross_att)
            self.num_heads.append(cross_num_heads)
            self.dim_head.append(cross_dim_head)
            set_pruning_att(model.get_submodule(module_name).cross_att, len(self.att)-1, self.att, self.num_heads, self.dim_head)

        # check exception
        if len(self.transformer) == 0:
            raise TypeError("plugin doesn't maintain any mask, all the mask lists are empty, \
                            please check if your model has the module type: bmt.CheckpointBlock or model_center.layer.TransformerBlock")
        elif any((len(self.ffn) == 0, len(self.att) == 0, len(self.num_heads) == 0, len(self.dim_head) == 0, len(self.dim_ff) == 0)):
            raise ValueError("Now BMCook doesn't support to prune model without feedforward layer or attention layer. It's also not allowed only layernorm parameters exist in these layers.")

        if os.path.exists(saved_path):
            self.load_masks(saved_path, model)
        else:
            self.set_pruning(model)

    def set_pruning(self, model: Module):
        for name, module in model.named_modules():
            if self.transformer.from_name(name) is not None:
                set_pruning_transformer(module, self.transformer.name2index(name), self.transformer)
            elif self.att.from_name(name) is not None:
                set_pruning_att(module, self.att.name2index(name), self.att, self.num_heads, self.dim_head)
            elif self.ffn.from_name(name):
                set_pruning_ffn(module, self.ffn.name2index(name), self.ffn, self.dim_ff)
    
    def masks_setup(self):
        base_grain = max([mask.grain for mask in self.training_masks.values()])
        top_grain = min([mask.grain for mask in self.training_masks.values()])
        self.start_name = []
        for name, mask in self.training_masks.items():
            if mask.grain == top_grain:
                self.start_name.append(name)
            if mask.grain == base_grain:
                self.training_masks[name].set_leaf()
            elif mask.grain > base_grain:
                self.training_masks[name].set_drop()
    
    def __len__(self):
        return len(self.transformer)

    def get_param_exp_all(self):
        # transformer = self.transformer[index]
        params_exp, params_all = 0, 0
        for name in self.start_name:
            mask = self.training_masks[name]
            for item in mask:
                params_exp += item.get_param_exp()
                params_all += item.get_param_all()
        
        return params_exp, params_all
    
    def get_masks(self, key: str = None):
        r"""print the masks managed in SPrunePlugin"""
        res = {'transformer': self.transformer.get_mask(),
                'att': self.att.get_mask(),
                'ffn': self.ffn.get_mask(),
                'num_heads': self.num_heads.get_mask(),
                'dim_head': self.dim_head.get_mask(),
                'dim_ff': self.dim_ff.get_mask()}
        if key is None:
            return res
        else:
            return res[key]

    def save_masks(self, path, all_masks: bool = False):
        r"""save the plugin as a dict.
        Args:
            path: `(str)`, the save path.
        """
        res = OrderedDict()
        for name, mask in self.training_masks.items():
            res[name] = mask.get_mask()
        if all_masks:
            for k, v in self.__dict__.items():
                if isinstance(v, UnitList):
                    res[k] = v.get_mask()
        torch.save(res, path)

    def load_masks(self, path, model=None):
        r"""load the saved dict to this plugin.
        Args:
            path: `(str)`, the file path.
        """
        masks_dict = torch.load(path)
        for k, _ in self.__dict__.items():
            if k in masks_dict:
                self.__dict__[k].load_mask(masks_dict[k])
        if model is not None:
            self.set_pruning(model)