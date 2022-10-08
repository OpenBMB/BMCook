from operator import mod
from symbol import factor
import torch
import bmtrain as bmt
from model_center.layer import TransformerBlock
from model_center.model import BaseModel, Config
from typing import Dict, Optional, Union, Tuple, Any, List
from collections import OrderedDict

from .utils import set_pruning_att, set_pruning_ffn, set_pruning_transformer, get_params_from_block

class SPruneStrategy:
    def __init__(self, config: Dict) -> None:
        main_config = config['main']
        target_mode_config = config['target_mode']
        iterational_config = config['iterational']
        
        self.criterion = main_config['criterion']
        assert self.criterion == 'l0', "BMCook sprune do not support other criterions besides l0 yet."
        self.fixed_mask_path = main_config['fixed_mask_path']
        self.training_mask = main_config['training_mask']
        self.mask_mode = main_config['mask_mode']
        self.target_mode = main_config['target_mode']
        self.iterational = main_config['iterational']

        if self.target_mode == 'dimention':
            self.target_dimention = target_mode_config['dimention']
        elif self.target_mode == 'sparsity':
            self.target_sparsity = target_mode_config['sparsity']

        if self.iterational is True:
            self.iter_interval = iterational_config['interval']
            self.iter_ratio = iterational_config['ratio']


class SPrunePlugin:
    '''
    SPrune is a base class for structure prune in BMCook.
    All the modules supported by Sprune includes: transformer block layer, attention layer, feedforward layer, num_heads, dim_head, dim_ff.

    '''
    def __init__(self, config: Dict, model: BaseModel):
        '''
        analyze the structure prune methed
        '''
        self._strategy = SPruneStrategy(config)
        
        # read some hyperparameters
        model_config = model.config
        dim_model = model_config.dim_model
        num_heads = model_config.num_heads
        dim_head = model_config.dim_head
        if 'num_layers' in model_config.__dict__:
            num_layers = model_config.num_layers
        elif 'num_encoder_layers' in model_config.__dict__:
            num_encoder_layers = model_config.num_encoder_layers
            num_decoder_layers = model_config.num_decoder_layers
        else:
            raise AttributeError("Missing num_layers or num_encoder_layers/num_decoder_layers in this config.")

        # model analysis
        prunable_all_params = 0
        self_att_num, cross_att_num = 0, 0
        TRANSFORMER_MASK, FFN_MASK, ATT_MASK = [], [], []
        NUM_HEADS_MASK, DIM_HEAD_MASK, DIM_FF_MASK = [], [], []
        for name, module in model.named_modules():
            if type(module) in (bmt.block_layer.CheckpointBlock, TransformerBlock):
                block_type, overall_index = name.split('.')[0], int(name.split('.')[2])
                if block_type == 'decoder':
                    overall_index += num_encoder_layers

                ordered_parameters = [(k, v) for k, v in module.state_dict().items()]
                self_att_param, cross_att_param, ffn_param = 0, 0, 0
                for (k, v) in ordered_parameters:
                    if 'self_att' in k:
                        self_att_param += v.numel()
                    elif 'cross_att' in k:
                        cross_att_param += v.numel()
                    elif 'ffn' in k:
                        ffn_param += v.numel()
                
                # model ststistics
                transformer_layer_param = self_att_param + cross_att_param + ffn_param
                prunable_all_params += transformer_layer_param

                if transformer_layer_param > 0:
                    TRANSFORMER_MASK.append({
                                            'index': overall_index,
                                            'param': transformer_layer_param,
                                            'mask': None
                                            })
                    set_pruning_transformer(module, len(TRANSFORMER_MASK)-1, TRANSFORMER_MASK)

                    if self_att_param > 0:
                        ATT_MASK.append({
                                                'index': overall_index, 
                                                'param': self_att_param,
                                                'mask': None,
                                                })
                        NUM_HEADS_MASK.append({
                                                'index': overall_index, 
                                                'param': dim_head,
                                                'mask': None
                                                })
                        DIM_HEAD_MASK.append({
                                                'index': overall_index, 
                                                'param': num_heads,
                                                'mask': None
                                                })
                        set_pruning_att(module.self_att, len(ATT_MASK)-1, ATT_MASK, NUM_HEADS_MASK, DIM_HEAD_MASK)
                        self_att_num += 1
                        
                    
                    if cross_att_param > 0:
                        ATT_MASK.append({
                                                'index': overall_index, 
                                                'param': cross_att_param,
                                                'mask': None,
                                                })
                        NUM_HEADS_MASK.append({
                                                    'index': overall_index, 
                                                    'param': dim_head,
                                                    'mask': None
                                                    })
                        DIM_HEAD_MASK.append({
                                                    'index': overall_index, 
                                                    'param': num_heads,
                                                    'mask': None
                                                    })
                        set_pruning_att(module.cross_att, len(ATT_MASK)-1, ATT_MASK, NUM_HEADS_MASK, DIM_HEAD_MASK)
                        cross_att_num += 1

                    if ffn_param > 0:
                        FFN_MASK.append({
                                        'index': overall_index, 
                                        'param': ffn_param,
                                        'mask': None,
                                        })
                        DIM_FF_MASK.append({
                                            'index': overall_index, 
                                            'param': dim_model,
                                            'mask': None
                                            })
                        set_pruning_ffn(module.ffn, len(FFN_MASK)-1, FFN_MASK, DIM_FF_MASK)
        
        # init mask shape for the use of loga
        transformer_mask_shape = (len(TRANSFORMER_MASK))
        
        num_heads_list = [mask['param'] for mask in NUM_HEADS_MASK]
        num_heads_num, max_num_heads = len(NUM_HEADS_MASK), max(num_heads_list)
        num_heads_shape = (num_heads_num, max_num_heads)
        num_heads_shape_mask = torch.stack(
            [(torch.arange(max_num_heads) < att['param']).long() \
                for att in NUM_HEADS_MASK]
            )
        
        dim_head_list = [mask['param'] for mask in DIM_HEAD_MASK]
        dim_head_num, max_dim_head = len(DIM_HEAD_MASK), max(dim_head_list)
        dim_head_shape = (dim_head_num, max_dim_head)
        dim_head_shape_mask = torch.stack(
            [(torch.arange(max_dim_head) < att['param']).long() \
                for att in DIM_HEAD_MASK]
        )
        
        ffn_list = [mask['param'] for mask in DIM_FF_MASK]
        ffn_num, max_dim_ff = len(DIM_FF_MASK), max(ffn_list)
        ffn_shape = (ffn_num, max_dim_ff)
        ffn_shape_mask = torch.stack(
            [(torch.arange(max_dim_ff) < ffn['param']).long() \
                for ffn in FFN_MASK]
        )

        self._info = {
            'all_params': prunable_all_params,
            'att_boundary': self_att_num - cross_att_num,
            'shape':{
                'transformer': ((transformer_mask_shape), torch.ones(transformer_mask_shape)),
                'att': ((num_heads_num), torch.ones(num_heads_num)),
                'ffn': ((ffn_num), torch.ones(ffn_num)),
                'num_heads': (num_heads_shape, num_heads_shape_mask),
                'dim_head': (dim_head_shape, dim_head_shape_mask),
                'dim_ff': (ffn_shape, ffn_shape_mask)
            }
        }

        self._transformer = TRANSFORMER_MASK
        self._att = ATT_MASK
        self._ffn = FFN_MASK
        self._num_heads = NUM_HEADS_MASK
        self._dim_head = DIM_HEAD_MASK
        self._fim_ff = DIM_FF_MASK

        del num_heads_list, dim_head_list, ffn_list

        self.setup_mask_for_plugin()

        model.sprune_plugin = self

    def setup_mask_for_plugin(self):
        '''
        function 'setup_prune_grain' is to set up the specific prune granularity.
        '''
        for mask in self._strategy.training_mask:
            assert mask in {"transformer", "att", "ffn", "num_heads", "dim_head", "dim_ff"}, \
                "BMCook Sprune only support training mask of (transformer_layer, self_att_layer, cross_att_layer, ffn_layer, num_heads, dim_head, dim_ff), but got {}".format(self.training_mask)
        
        # train mask
        pop_list = []
        for k in list(self._info['shape']):
            if k not in self._strategy.training_mask:
                pop_list.append(k)
        for k in pop_list:
            self._info['shape'].pop(k)

        # fix mask
        if self._strategy.fixed_mask_path != "":
            fixed_masks = torch.load(self._strategy.fixed_mask_path)
            assert type(fixed_masks) == dict, "the fixed mask should be dictionary"

            # the fixed mask should be 0 or 1
            for k, v in fixed_masks.items():  # k: name, v: list
                assert len(self.__dict__[k]) == v.size(0)
                for i in range(v.size(0)):
                    self.__dict__[k][i]['mask'] = v[i]
                self._info['shape'].pop(k)

    def training_masks(self):
        '''
        traverse all the necessary masks in this training.
        '''
        for name in self._info['shape'].keys():
            for i, v in enumerate(self.__dict__['_'+name]):
                if v['mask'] is not None:
                    yield name + '.' + str(i)
    
    def training_modules(self):
        '''
        traverse all the necessary modules in this training.
        '''
        for name in self._info['shape'].keys():
            for _, v in enumerate(self.__dict__['_'+name]):
                if v['mask'] is not None:
                    yield name
                    break

    def get_sparsity(self):
        '''
        calculate the sparsity in single grain
        '''
        all_params, expected_params = 0, 0
        info_list = {}

        for name in self.training_masks():
            module, index = name.split('.')[0], int(name.split('.')[1])
            param = self.__dict__['_' + module][index]['param']
            mask = self.__dict__['_' + module][index]['mask']
            index = self.__dict__['_' + module][index]['index']
            if index not in info_list:
                info_list[index] = {'module': [module], 'param': [param], 'score': [mask]}
            else:
                if module in info_list[index]['module']:
                    module = 'cross_' + module
                info_list[index]['module'].append(module)
                info_list[index]['param'].append(param)
                info_list[index]['score'].append(mask)
        
        expected_params, all_params = get_params_from_block(info_list)

        sparsity = 1 - expected_params / all_params
        return sparsity