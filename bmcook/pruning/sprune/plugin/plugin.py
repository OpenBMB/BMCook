import torch
from typing import Dict, List, Optional, Any
from collections import OrderedDict
import bmtrain as bmt

from .unit import TransformerUnit, SPruneUnit, AttUnit, FFNUnit
from .unitlist import UnitList


def _process_mask_names(mask_names: List[str]):
    module_masks, param_masks = [], []
    for name in mask_names:
        if name in ["num_heads", "dim_head", "dim_model"]:
            module_masks.append("att_params")
            param_masks.append(name)
        elif name in ["dim_ff", "dim_model"]:
            module_masks.append("ffn_params")
            param_masks.append(name)
        else:
            module_masks.append(name)
    return list(set(module_masks)), param_masks


class SPrunePlugin:
    r"""
    SPrunePlugin is a base class for structure prune in BMCook.
    All the modules supported by SprunePlugin includes: TransformerBlock layer, Attention layer, Feedforward layer, num_heads, dim_head, dim_ff.
    """
    def __init__(self, 
        mask_names: List[str], 
        model: torch.nn.Module, 
        from_model: bool = True
        ) -> None:
        '''
        analyze the structure prune methed.
        _module_masks: store the granularity which will be pruned. Give a row-view.
        _top_masks: store the top level sprune unit, i.e. transformerunit. Give a column-view.
        '''
        self.__dict__["_top_masks"]: Dict[str, SPruneUnit] = OrderedDict()
        self.__dict__["_module_masks"]: Dict[str, UnitList] = OrderedDict()
        self.__dict__["_param_masks"]: Dict[str, UnitList] = OrderedDict()

        module_masks, param_masks = _process_mask_names(mask_names)
        
        # add row-view interface.
        for mask_name in module_masks:
            setattr(self, mask_name, UnitList())
        
        for mask_name in param_masks:
            _param_masks = self.__dict__.get("_param_masks")
            _param_masks[mask_name] = UnitList()

        # add column-view interface.
        if from_model:
            for name, module in model.encoder.layers.named_children():
                if not (hasattr(module, 'self_att') or hasattr(module, 'ffn')):
                    continue
                prefix = 'encoder.layers.'
                setattr(self, prefix+name, TransformerUnit(module, prefix+name))
        else:
            self.from_config()

        # set pruning according to row-view interface, to support granularity-based pruning.
        for unitlist in self._module_masks.values():
            for unit in unitlist:
                module = model.get_submodule(unit.name)
                unit.set_pruning(module)

    def from_config(config):
        pass

    def __iter__(self):
        for v in self._top_masks.values():
            yield v
    
    def __getitem__(self, __i: int):
        for i, v in enumerate(self._top_masks.values()):
            if i == __i:
                return v

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, UnitList):
            mask_list = self.__dict__.get('_module_masks')
            mask_list[__name] = __value
        elif isinstance(__value, SPruneUnit):
            masks = self.__dict__.get('_top_masks')
            mask_list = self.__dict__.get('_module_masks')
            param_list = self.__dict__.get('_param_masks')
            
            # add column-view interface
            if isinstance(__value, TransformerUnit):
                masks[__name] = __value
            # put active unit into unitlist
            list_name = __value.list_in
            if list_name in mask_list:
                __value.set_training()
                mask_list[list_name].append(__value)
            elif list_name in param_list:
                __value.set_training()
                param_list[list_name].append(__value)

            for v in __value.__dict__.values():
                if isinstance(v, SPruneUnit):
                    self.__setattr__(v.name, v)
    
    def __getattr__(self, name):
        _top_masks = self.__dict__['_top_masks']
        _module_masks = self.__dict__['_module_masks']
        _param_masks = self.__dict__["_param_masks"]
        if name in _top_masks:
            return _top_masks[name]
        elif name in _module_masks:
            return _module_masks[name]
        elif name in _param_masks:
            return _param_masks[name]
  
    def __delattr__(self, __name: str) -> None:
        if __name in self._top_masks:
            for n, mask in self._top_masks[__name].__dict__.items():
                if isinstance(mask, SPruneUnit):
                    list_in = self._module_masks[mask.list_in]
                    self._module_masks[mask.list_in].pop(list_in.name2index(n))
            del self._top_masks[__name]
        elif __name in self._module_masks:
            del self._module_masks[__name]
        elif __name in self._param_masks:
            del self._param_masks[__name]

    def named_lists(self):
        for name, unilist in self._module_masks.items():
            if "params" in name:
                continue
            yield name, unilist
            
        for name, unilist in self._param_masks.items():
            yield name, unilist

    def prunable_name_lists(self):
        for name, unilist in self._module_masks.items():
            yield name, unilist

    def iter_namsed_masks(self):
        for n, mask in self._top_masks.items():
            yield n, mask

    def save_masks(self, path):
        r"""save the plugin as a dict.
        Args:
            path: `(str)`, the save path.
        """
        res = OrderedDict()
        for name, mask in self.named_lists():
            res[name] = mask.get_mask()
        torch.save(res, path)
