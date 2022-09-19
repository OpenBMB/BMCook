from turtle import forward
import types
import torch
import bmtrain as bmt
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Dict
from ..utils.config import ConfigParser

from model_center.layer import TransformerBlock, FeedForward, Attention
from sprune_func import cdf_concrete_dist, sample_mask, binarize_mask

def create_sprune_optimizer(model : nn.Module):
    l0_params = [{
                "params": [p for n, p in model.named_parameters() if "lambda" not in n],
                "weight_decay": 0.0,
                "lr": 0.1
            }]
    sp_optimizer = torch.optim.AdamW(l0_params)

    lagrangian_params = [{
                        "params": [p for n, p in model.named_parameters() if "lambda" in n],
                        "weight_decay": 0.0,
                        "lr": -0.1
                        }]
    lagrangian_optimizer = torch.optim.AdamW(lagrangian_params)

    return (sp_optimizer, lagrangian_optimizer)

MASK_GRANURALITY = {
    0: ['transformer_layer'],
    1: ['self_att_layer', 'cross_att_layer', 'ffn_layer'],
    2: ['num_heads', 'dim_head', 'dim_ff']
}

def set_pruning_in_forward(func_old):
    def _func_new(x, mask):
        out_old = func_old(x)
        out_new = out_old * mask
        return out_new
    return _func_new

class SPrunePlugin:
    '''
    SPrunePlugin is a base class for structure prune in BMCook.
    All the modules supported by Sprune includes: transformer block layer, attention layer, feedforward layer, num_heads, dim_head, dim_ff.
    '''
    def __init__(self, config : ConfigParser) -> None:
        '''
        analyze the structure prune methed
        '''
        main_config = config.get('main')
        target = config.get('target_mode')
        
        self.criterion = main_config['criterion']
        assert self.criterion == 'l0', "BMCook sprune do not support other criterions besides l0 yet."
        
        self.training_mask = main_config['training_mask']
        self.fixed_mask_path = main_config['fixed_mask_path']
        self.mask_method = main_config['mask_method']
        self.mask_mode = main_config['mask_mode']
        self.target_mode = main_config['target_mode']
        self.iterational = main_config['iterational']

        if self.target_mode == 'dimention':
            self.target_dimention = target['dimention']
        elif self.target_mode == 'sparsity':
            self.target_sparsity = target['sparsity']
    
    def setup_target_model(self, model : nn.Module, model_config):
        # read some hyperparameters
        self.dim_model = model_config.dim_model
        self.dim_ff = model_config.dim_ff
        self.num_heads = model_config.num_heads
        self.dim_head = model_config.dim_head
        if 'num_layers' in model_config.__dict__:
            self.num_layers = model_config.num_layers
        elif 'num_encoder_layers' in model_config.__dict__:
            self.num_encoder_layers = model_config.num_encoder_layers
            self.num_decoder_layers = model_config.num_decoder_layers
        else:
            raise AttributeError("Missing num_layers or num_encoder_layers/num_decoder_layers in this config.")

        # model analysis -- in att/ffn grain
        self.prunable_all_params = 0
        self.self_att_mapping, self.cross_att_mapping, self.ffn_mapping = {}, {}, {}
        self_att_index, cross_att_index, ffn_index = 0, 0, 0
        for name, module in model.named_modules():
            if type(module) in (bmt.block_layer.CheckpointBlock, TransformerBlock):
                block_type, transformer_index = name.split('.')[0], int(name.split('.')[2])
                if block_type == 'decoder':
                    transformer_index += self.num_encoder_layers
                ordered_parameters = [(k, v) for k, v in module.state_dict().items()]
                self_att_param, cross_att_param, ffn_param = 0, 0, 0
                for (k, v) in ordered_parameters:
                    if 'self_att' in k:
                        self_att_param += v.numel()
                    elif 'cross_att' in k:
                        cross_att_param += v.numel()
                    elif 'ffn' in k:
                        ffn_param += v.numel()
                
                if self_att_param > 0:
                    self.self_att_mapping[transformer_index] = {
                                                                'att_index': self_att_index, 
                                                                'param': self_att_param,
                                                                'num_heads': self.num_heads,
                                                                'dim_head': self.dim_head
                                                                }
                    self_att_index += 1
                    self.prunable_all_params += self_att_param
                else:
                    self.self_att_mapping[transformer_index] = None
                
                if cross_att_param > 0:
                    self.cross_att_mapping[transformer_index] = {
                                                                'att_index': cross_att_index, 
                                                                'param': cross_att_param,
                                                                'num_heads': self.num_heads,
                                                                'dim_head': self.dim_head
                                                                }
                    cross_att_index += 1
                    self.prunable_all_params += cross_att_param
                else:
                    self.cross_att_mapping[transformer_index] = None
                
                if ffn_param > 0:
                    self.ffn_mapping[transformer_index] = {
                                                            'ffn_index': ffn_index, 
                                                            'param': ffn_param,
                                                            'dim_ff': self.dim_ff
                                                            }
                    ffn_index += 1
                    self.prunable_all_params += ffn_param
                else:
                    self.ffn_mapping[transformer_index] = None
        self.num_self_att_layers = self_att_index
        self.num_cross_att_layers = cross_att_index
        self.num_ffn_layers = ffn_index

    def setup_prune_granularity(self):
        '''
        function 'setup_prune_grain' is to set up the specific prune granularity.
        '''
        assert self.training_mask in {"transformer_layer", "self_att_layer", "cross_att_layer", "ffn_layer", "num_heads", "dim_head", "dim_ff"}, \
                "BMCook.Sprune only support training mask of (transformer_layer, self_att_layer, cross_att_layer, ffn_layer, num_heads, dim_head, dim_ff), but got {}".format(self.training_mask)
        
        if self.fixed_mask_path != "":
            fixed_masks = torch.load(self.fixed_mask_path)
            assert type(fixed_masks) == dict, "the fixed mask should be dictionary"

        dct = self.__dict__
        for k, v in fixed_masks.items():
            dct[k+'_z'] = Parameter(v, requires_grad=False)

        MASK_SHPAE_MAPPING = {
            "transformer_layer": (self.num_layers),
            "self_att_layer": (self.num_self_att_layers),
            "cross_att_layer": (self.num_cross_att_layers),
            "ffn_layer": (self.num_ffn_layers),
            "dim_ff": (self.num_ffn_layers, self.dim_ff),
            "num_heads": (self.num_self_att_layers + self.num_cross_att_layers, self.num_heads),
            "dim_head": (self.num_self_att_layers + self.num_cross_att_layers, self.dim_head),
        }

        training_loga = {}
        for mask in self.training_mask:
            training_loga[mask+'_loga'] = Parameter(torch.empty(MASK_SHPAE_MAPPING[mask], dtype=torch.float))

    def setup_forward_for_pruning(self, model : nn.Module, mask):
        # change model forward
        for name, module in model.named_modules():
            if 'ffn.w_in' in name:
                module.forward_unprune = module.forward
                def _new_forward(module_self, x, dim_ff_z):
                    out = module_self.forward_unprune(x)
                    if dim_ff_z is not None:
                        out = out * dim_ff_z
                    return out
                module.forward = types.MethodType(_new_forward, module)
            elif 'ffn.w_out' in name:
                module.forward_unprune = module.forward
                def _new_forward(module_self, x, dim_ff_z):
                    if dim_ff_z is not None:
                        x = x * dim_ff_z
                    out = module_self.forward_unprune(x)
                    return out
                module.forward = types.MethodType(_new_forward, module)
            elif 'self_attention.project' in name:
                module.forward_unprune = module.forward
                def _new_forward(module_self, x, num_heads_z, dim_head_z):
                    out = module_self.forward_unprune(x)  # (batch, len, num_heads*dim_head)
                    if num_heads_z is not None:
                        old_size = out.size()
                        out = out.view(old_size[0], old_size[1], self.num_heads, self.dim_head)
                        out = out * num_heads_z.view(self.num_heads, 1)
                        out = out.view(old_size[0], old_size[1], self.num_heads * self.dim_head)
                    if dim_head_z is not None:
                        old_size = out.size()
                        out = out.view(old_size[0], old_size[1], self.num_heads, self.dim_head)
                        out = out * dim_head_z
                        out = out.view(old_size[0], old_size[1], self.num_heads * self.dim_head)
                    return out
                module.forward = types.MethodType(_new_forward, module)
            elif 'self_attention.attention_out' in name:
                module.forward_unprune = module.forward
                def _new_forward(module_self, x, num_heads_z, dim_head_z):
                    if num_heads_z is not None:
                        old_size = x.size()  # (batch, len, num_heads * dim_head)
                        x = x.view(old_size[0], old_size[1], self.num_heads, self.dim_head)
                        x = x * num_heads_z.view(self.num_heads, 1)
                        x = x.view(old_size[0], old_size[1], self.num_heads * self.dim_head)
                    if dim_head_z is not None:
                        old_size = x.size()
                        x = x.view(old_size[0], old_size[1], self.num_heads, self.dim_head)
                        x = x * dim_head_z
                        x = x.view(old_size[0], old_size[1], self.num_heads * self.dim_head)
                    out = module_self.forward_unprune(x)  # (batch, len, dim_model)
                    return out
                module.forward = types.MethodType(_new_forward, module)
            elif type(module) == FeedForward:
                module.forward_unprune = module.forward
                def _new_forward(module_self, x, ffn_layer_z):
                    out = module_self.forward_unprune(x)
                    out = out * ffn_layer_z
                    return out
                module.forward = types.MethodType(_new_forward, module)
            elif type(module) == Attention:
                module.forward_unprune = module.forward
                def _new_forward(module_self, x, att_layer_z):
                    out = module_self.forward_unprune(x)
                    out = out * att_layer_z
                    return out
                module.forward = types.MethodType(_new_forward, module)
            elif type(module) == TransformerBlock:
                #TODO deve transformer level pruning
                pass

        pass


class SPruneEngine:
    def __init__(self, training_masks : Dict) -> None:
        super().__init__()
        self.lambda_1 = Parameter(torch.tensor(0.0, dtype=torch.float, device='cuda'))
        self.lambda_2 = Parameter(torch.tensor(0.0, dtype=torch.float, device='cuda'))

        self.training_loga = {}
        for mask, shape in training_masks.items():
            self.training_loga[mask+'_loga'] = Parameter(torch.empty(shape, dtype=torch.float).normal_(10., 1e-2))

    def forward(self, training : bool = True):
        masks = {}
        for k, v in self.training_loga.items():
            prefix = k.split('_')[0]
            masks[prefix+'_mask'] = sample_mask(v) if training is True else binarize_mask(v)
        
        return masks

    def get_num_parameters(self):
        pass

    def lagrangian_loss_sparsity(self):
        expected_size = self.get_num_parameters()
        
        expected_sparsity = 1 - expected_size / self.prunable_all_params
        loss_sparsity = expected_sparsity - self.target_sparsity

        lagrangian_loss = self.lambda_1 * loss_sparsity + self.lambda_2 * (loss_sparsity ** 2)

        return lagrangian_loss, expected_sparsity

    def lagrangian_loss_dimension(self):
        '''
        target mode maske sense
        '''
        dimension_score = 1 - cdf_concrete_dist(torch.tensor(0.), self.training_loga).cuda()
        all_dimension = dimension_score.size(1)
        
        expected_dimension = torch.sum(dimension_score, -1)
        loss_dimension = torch.sum((self.target_dimension - expected_dimension) / all_dimension)
        
        lagrangian_loss = self.lambda_1 * loss_dimension + self.lambda_2 * (loss_dimension ** 2)
        
        return lagrangian_loss, expected_dimension
        if self.prune_mode == 'att':
            target_num_heads = self.target_num_heads
            heads_score = 1 - self.cdf_concrete_dist(torch.tensor(0.), self.heads_loga).cuda()
            expected_num_heads = torch.sum(heads_score, -1)
            loss_heads = torch.sum((target_num_heads - expected_num_heads) / self.num_heads)
            lagrangian_loss = self.lambda_1 * (loss_heads) + \
                                self.lambda_2 * (loss_heads) ** 2
            expected_num = expected_num_heads
            target = target_num_heads
        
        elif self.prune_mode == 'ffn':
            target_dimff = self.target_dimff
            dimff_score = 1 - self.cdf_concrete_dist(torch.tensor(0.), self.dimff_loga).cuda()
            expected_num_dimff = torch.sum(dimff_score, -1)
            loss_dimff = torch.sum((target_dimff - expected_num_dimff) / self.dim_ff)
            lagrangian_loss = self.lambda_1 * (loss_dimff) + \
                                self.lambda_2 * (loss_dimff) ** 2
            expected_num = expected_num_dimff
            target = target_dimff

        return lagrangian_loss, expected_num, target


class L0_Module_coarse(SPruneEngine):
    '''
    This L0_Module_coarse is for layer mask training
    '''
    def __init__(self, 
                model : bmt.CheckpointBlock,
                config
                ) -> None:
        super(L0_Module_coarse, self).__init__()
        self.target_sparsity = config['target_sparsity']

        self.masks = []
        ordered_parameters = []
        self.num_self_att_layer = 0
        self.num_cross_att_layer = 0
        self.num_ffn_layer = 0
        self.prunable_all_params = 0
        self.params_per_self_att = 0
        self.params_per_cross_att = 0
        self.params_per_ffn = 0

        for prefix, block in model.named_modules():
            if bmt.block_layer.CheckpointBlock == type(block):
                block_for_info = block
                ordered_parameters = [(k, v) for k, v in block.state_dict().items()]

                self.prunable_all_params += sum([param['size'] for param in block._param_info if 'layernorm' not in param['name']])

                for (k, v) in ordered_parameters:
                    if 'self_att' in k:
                        self.num_self_att_layer += 1
                        break
                for (k, v) in ordered_parameters:
                    if 'ffn' in k:
                        self.num_ffn_layer += 1
                        break
                if 'encoder' in prefix:
                    pass
                else:
                    self.num_cross_att_layer += 1

        for param in block_for_info._param_info:
            if 'self_att' in param['name']:
                self.params_per_self_att += param['size']
            elif 'cross_att' in param['name']:
                self.params_per_cross_att += param['size']
            elif 'ffn' in param['name']:
                self.params_per_ffn += param['size']
        
        self.self_att_layer_loga = self.initialize_parameters((self.num_self_att_layer))
        self.cross_att_layer_loga = self.initialize_parameters((self.num_cross_att_layer))
        self.ffn_layer_loga = self.initialize_parameters((self.num_ffn_layer))

    def _binarize_mask(self, loga, size):
        expected_num_nonzeros = torch.sum(1 - cdf_concrete_dist(torch.tensor(0.), loga))
        expected_num_zeros = size - expected_num_nonzeros.item()
        num_zeros = round(expected_num_zeros)
        soft_mask = torch.ones_like(loga)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(loga, k=num_zeros, largest=False)  # 返回values, indices
                soft_mask[indices] = 0.  # 置零
        return soft_mask

    def forward(self, training=True):
        zs = {"self_att_layer_z": [], 
                "cross_att_layer_z": [],
                "ffn_layer_z": []}

        if training:
            bmt.print_rank(self.ffn_layer_loga)
            zs["self_att_layer_z"] = sample_mask(self.self_att_layer_loga).reshape((self.num_self_att_layer))
            zs["cross_att_layer_z"] = sample_mask(self.cross_att_layer_loga).reshape((self.num_cross_att_layer))
            zs["ffn_layer_z"] = sample_mask(self.ffn_layer_loga).reshape((self.num_ffn_layer))
        else:
            zs["self_att_layer_z"] = self._binarize_mask(self.self_att_layer_loga, self.num_self_att_layer).reshape((self.num_self_att_layer))
            zs["cross_att_layer_z"] = self._binarize_mask(self.cross_att_layer_loga, self.num_cross_att_layer).reshape((self.num_cross_att_layer))
            zs["ffn_layer_z"] = self._binarize_mask(self.ffn_layer_loga, self.num_ffn_layer).reshape((self.num_ffn_layer))
        return zs

    def get_num_parameters(self):
        num_parameters = 0

        self_att_score = 1 - cdf_concrete_dist(torch.tensor(0.), self.self_att_layer_loga)
        cross_att_score = 1 - cdf_concrete_dist(torch.tensor(0.), self.cross_att_layer_loga)
        ffn_score = 1 - cdf_concrete_dist(torch.tensor(0.), self.ffn_layer_loga)
        
        num_parameters += torch.sum(self_att_score) * self.params_per_self_att
        num_parameters += torch.sum(cross_att_score) * self.params_per_cross_att
        num_parameters += torch.sum(ffn_score) * self.params_per_ffn

        return num_parameters

    def lagrangian_regularization(self):
        '''
        target mode make sense
        '''
        target_sparsity = self.target_sparsity  # 指定目标sparsity
        expected_size = self.get_num_parameters()
        
        expected_sparsity = 1 - expected_size / self.prunable_all_params

        lagrangian_loss = self.lambda_1 * (expected_sparsity - target_sparsity) + \
                            self.lambda_2 * (expected_sparsity - target_sparsity) ** 2

        return lagrangian_loss, expected_sparsity, target_sparsity

class L0_Module_fine(SPruneEngine):
    '''
    This L0_Module_fine is for heads/dimff mask training given fixed layer mask
    '''
    def __init__(self, 
                prune_config,
                ):
        super(L0_Module_fine, self).__init__()
        self.layer_zs = torch.load(prune_config['coarse_mask'])
        self.num_heads = prune_config['num_heads']
        self.dim_ff = prune_config['dim_ff']
        self.prune_mode = prune_config['prune_mode']
        
        self.att_zs = self.layer_zs['self_att_layer_z']
        self.ffn_zs = self.layer_zs['ffn_layer_z']
        self.num_att_layers = int(torch.sum(self.att_zs).item())
        self.num_ffn_layers = int(torch.sum(self.ffn_zs).item())

        if self.prune_mode == 'att':
            self.heads_loga = self.initialize_parameters((self.num_att_layers, self.num_heads))
            self.target_num_heads = prune_config['target_num_heads']
        elif self.prune_mode == 'ffn':
            mask_z = torch.load(prune_config['heads_mask'])
            heads_z = mask_z['heads_z']  # (25, 32)
            self.heads_loga = Parameter(heads_z, requires_grad=False)
            self.dimff_loga = self.initialize_parameters((self.num_ffn_layers, self.dim_ff))
            self.target_dimff = prune_config['target_dimff']

    def _binarize_mask(self, loga, size):
        expected_num_nonzeros = torch.sum(1 - self.cdf_concrete_dist(torch.tensor(0.), loga), -1)
        expected_num_zeros = size - expected_num_nonzeros
        res, num_zeros = [], []
        if self.prune_mode == 'att':
            num_zeros = self.num_heads - self.target_num_heads
        elif self.prune_mode == 'ffn':
            num_zeros = self.dim_ff - self.target_dimff
        for index in range(len(expected_num_zeros)):
            cur_layer = loga[index]
            soft_mask = torch.ones_like(cur_layer)
            if num_zeros > 0:
                if soft_mask.ndim == 0:
                    soft_mask = torch.tensor(0).to(cur_layer.device)
                else:
                    _, indices = torch.topk(cur_layer, k=num_zeros, largest=False)  # 返回values, indices
                    soft_mask[indices] = 0.  # 置零
            res.append(soft_mask)
        return torch.stack(res)

    def forward(self, training=True):
        if self.prune_mode == 'att':
            zs = {"heads_z" : self.heads_loga}
            if training:
                zs["heads_z"] = sample_mask(self.heads_loga).reshape((self.num_att_layers, self.num_heads))
            else:
                zs["heads_z"] = self._binarize_mask(self.heads_loga, 25).reshape((self.num_att_layers, self.num_heads))

        elif self.prune_mode == 'ffn':
            zs = {"heads_z" : self.heads_loga, "dimff_z": self.dimff_loga,}
            if training:
                zs["dimff_z"] = sample_mask(self.dimff_loga).reshape((self.num_ffn_layers, self.dim_ff))
            else:
                zs["dimff_z"] = self._binarize_mask(self.dimff_loga, self.num_ffn_layers).reshape((self.num_ffn_layers, self.dim_ff))
        
        return zs

    def lagrangian_regularization(self):
        '''
        target mode maske sense
        '''
        if self.prune_mode == 'att':
            target_num_heads = self.target_num_heads
            heads_score = 1 - self.cdf_concrete_dist(torch.tensor(0.), self.heads_loga).cuda()
            expected_num_heads = torch.sum(heads_score, -1)
            loss_heads = torch.sum((target_num_heads - expected_num_heads) / self.num_heads)
            lagrangian_loss = self.lambda_1 * (loss_heads) + \
                                self.lambda_2 * (loss_heads) ** 2
            expected_num = expected_num_heads
            target = target_num_heads
        
        elif self.prune_mode == 'ffn':
            target_dimff = self.target_dimff
            dimff_score = 1 - self.cdf_concrete_dist(torch.tensor(0.), self.dimff_loga).cuda()
            expected_num_dimff = torch.sum(dimff_score, -1)
            loss_dimff = torch.sum((target_dimff - expected_num_dimff) / self.dim_ff)
            lagrangian_loss = self.lambda_1 * (loss_dimff) + \
                                self.lambda_2 * (loss_dimff) ** 2
            expected_num = expected_num_dimff
            target = target_dimff

        return lagrangian_loss, expected_num, target

