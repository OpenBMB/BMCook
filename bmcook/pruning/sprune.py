from unittest import expectedFailure
import torch
import bmtrain as bmt
import torch.nn as nn
from torch.nn.parameter import Parameter

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class L0_Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lambda_1 = Parameter(torch.tensor(0.0, dtype=torch.float, device='cuda'))
        self.lambda_2 = Parameter(torch.tensor(0.0, dtype=torch.float, device='cuda'))

    def _sample_z(self, loga, beta=2./3.):
        eps = torch.FloatTensor(*loga.shape).uniform_(epsilon, 1-epsilon).to(loga.device)
        s = torch.sigmoid((torch.log(eps / (1 - eps)) + loga) / beta)
        s = s * (limit_b - limit_a) + limit_a
        z = s.clamp(min=0., max=1.)
        return z
    
    def cdf_qz(self, eps, loga, beta=2./3.):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (eps - limit_a) / (limit_b - limit_a)  # 0.083333
        s = torch.sigmoid((torch.log(xn / (1 - xn)) * beta - loga))
        z = s.clamp(min=epsilon, max=1 - epsilon)
        return z

    def initialize_parameters(self, shape, dtype=torch.float, init_mean=10., init_std=1e-2, requires_grad=True):
        param = Parameter(torch.empty(shape, dtype=dtype), requires_grad=requires_grad)
        param.data.normal_(init_mean, init_std)
        param.requires_grad = True
        
        return param
    
    def create_sprune_optimizer(self):
        l0_params = [{
                    "params": [p for n, p in self.named_parameters() if "lambda" not in n],
                    "weight_decay": 0.0,
                    "lr": 0.1
                }]
        sp_optimizer = torch.optim.AdamW(l0_params)

        lagrangian_params = [{
                            "params": [p for n, p in self.named_parameters() if "lambda" in n],
                            "weight_decay": 0.0,
                            "lr": -0.1
                            }]
        lagrangian_optimizer = torch.optim.AdamW(lagrangian_params)

        return (sp_optimizer, lagrangian_optimizer)


class L0_Module_coarse(L0_Module):
    '''
    This L0_Module is for layer mask training
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

    def _deterministic_z(self, loga, size):
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(torch.tensor(0.), loga))
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
            zs["self_att_layer_z"] = self._sample_z(self.self_att_layer_loga).reshape((self.num_self_att_layer))
            zs["cross_att_layer_z"] = self._sample_z(self.cross_att_layer_loga).reshape((self.num_cross_att_layer))
            zs["ffn_layer_z"] = self._sample_z(self.ffn_layer_loga).reshape((self.num_ffn_layer))
        else:
            zs["self_att_layer_z"] = self._deterministic_z(self.self_att_layer_loga, self.num_self_att_layer).reshape((self.num_self_att_layer))
            zs["cross_att_layer_z"] = self._deterministic_z(self.cross_att_layer_loga, self.num_cross_att_layer).reshape((self.num_cross_att_layer))
            zs["ffn_layer_z"] = self._deterministic_z(self.ffn_layer_loga, self.num_ffn_layer).reshape((self.num_ffn_layer))
        return zs

    def get_num_parameters(self):
        num_parameters = 0

        self_att_score = 1 - self.cdf_qz(torch.tensor(0.), self.self_att_layer_loga)
        cross_att_score = 1 - self.cdf_qz(torch.tensor(0.), self.cross_att_layer_loga)
        ffn_score = 1 - self.cdf_qz(torch.tensor(0.), self.ffn_layer_loga)
        
        num_parameters += torch.sum(self_att_score) * self.params_per_self_att
        num_parameters += torch.sum(cross_att_score) * self.params_per_cross_att
        num_parameters += torch.sum(ffn_score) * self.params_per_ffn

        return num_parameters

    def lagrangian_regularization(self):
        target_sparsity = self.target_sparsity  # 指定目标sparsity
        expected_size = self.get_num_parameters()
        
        expected_sparsity = 1 - expected_size / self.prunable_all_params

        lagrangian_loss = self.lambda_1 * (expected_sparsity - target_sparsity) + \
                            self.lambda_2 * (expected_sparsity - target_sparsity) ** 2

        return lagrangian_loss, expected_sparsity, target_sparsity


class L0_Module_fine(L0_Module):
    '''
    This L0_Module_cofi is for heads/dimff mask training given fixed layer mask
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

    def _deterministic_z(self, loga, size):
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(torch.tensor(0.), loga), -1)
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
                zs["heads_z"] = self._sample_z(self.heads_loga).reshape((self.num_att_layers, self.num_heads))
            else:
                zs["heads_z"] = self._deterministic_z(self.heads_loga, 25).reshape((self.num_att_layers, self.num_heads))

        elif self.prune_mode == 'ffn':
            zs = {"heads_z" : self.heads_loga, "dimff_z": self.dimff_loga,}
            if training:
                zs["dimff_z"] = self._sample_z(self.dimff_loga).reshape((self.num_ffn_layers, self.dim_ff))
            else:
                zs["dimff_z"] = self._deterministic_z(self.dimff_loga, self.num_ffn_layers).reshape((self.num_ffn_layers, self.dim_ff))
        
        return zs

    def lagrangian_regularization(self):
        if self.prune_mode == 'att':
            target_num_heads = self.target_num_heads
            heads_score = 1 - self.cdf_qz(torch.tensor(0.), self.heads_loga).cuda()
            expected_num_heads = torch.sum(heads_score, -1)
            loss_heads = torch.sum((target_num_heads - expected_num_heads) / self.num_heads)

            lagrangian_loss = self.lambda_1 * (loss_heads) + \
                                self.lambda_2 * (loss_heads) ** 2
            expected_num = expected_num_heads
            target = target_num_heads
        
        elif self.prune_mode == 'ffn':
            target_dimff = self.target_dimff
            dimff_score = 1 - self.cdf_qz(torch.tensor(0.), self.dimff_loga).cuda()
            expected_num_dimff = torch.sum(dimff_score, -1)
            loss_dimff = torch.sum((target_dimff - expected_num_dimff) / self.dim_ff)
            lagrangian_loss = self.lambda_1 * (loss_dimff) + \
                                self.lambda_2 * (loss_dimff) ** 2
            expected_num = expected_num_dimff
            target = target_dimff

        return lagrangian_loss, expected_num, target

