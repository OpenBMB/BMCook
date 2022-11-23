import torch
import bmtrain as bmt
from typing import Dict
from torch.nn.parameter import Parameter
from .func import determinate_mask, sample, binarize, from_sparsity
from .plugin import SPrunePlugin, UnitList


class SPruneStrategy:
    def __init__(self, config: Dict) -> None:
        self.criterion = config['criterion']
        assert self.criterion == 'l0', "BMCook sprune do not support other criterions besides l0 yet."
        self.mask_path = config['mask_path']
        self.training_mask = config['training_mask']
        self.mode = config['mode']
        self.target_mode = config['target_mode']
        self.target_sparsity = config['target_sparsity']
        self.start_sparsity = config['start_sparsity']
        self.hard_binarize = config['hard_binarize']
        


class SPruneEngine(torch.nn.Module):
    r"""
    SPruneEngine is used for the mask computation and update of SPrunePlugin.

    The engine design is based on L0 regularization method and a lagrangian term. For L0 regularization details, see paper
        "Learning Sparse Neural Networks through L_0 Regularization" <https://openreview.net/forum?id=H1Y8hhg0b>.
        For lagrangian term in PLM structure pruning, see paper "Structured Pruning of Large Language Models" 
        <https://arxiv.org/abs/1910.04732>.
    """
    def __init__(self, config: Dict, plugin: SPrunePlugin) -> None:
        r"""Init the SpruneEngine from a SPrunePlugin. It will initilize all the :class:`torch.nn.Parameter`
        used for learning the sprune mask, and create the optimizer for l0 regularization.

        Args:
            config: `(Dict)`, the sprune config.
            plugin: `(SPrunePlugin)`, the SPrunePlugin.
        """
        super().__init__()
        self.strategy = SPruneStrategy(config)
        self.target_sparsity = self.strategy.target_sparsity
        self.mode = self.strategy.mode
        self.plugin = plugin
        self.training = True
        if self.mode == 'tuning':
            return 

        start_loga = from_sparsity(self.strategy.start_sparsity)

        self.lambda_1 = Parameter(torch.tensor(0., dtype=torch.float, device='cuda'))
        self.lambda_2 = Parameter(torch.tensor(0., dtype=torch.float, device='cuda'))
        # self.training_loga = {}
        for mask in self.strategy.training_mask:
            unitlist: UnitList = getattr(self.plugin, mask)
            setattr(self, mask+'_loga', Parameter(torch.empty((len(unitlist), unitlist.get_max_dim()), dtype=torch.float, device='cuda').normal_(start_loga, 1e-2)))
            self.plugin.training_masks[mask] = unitlist
        self.plugin.masks_setup()

        self.create_sprune_optimizer()

    def create_sprune_optimizer(self):
        r"""Create the sprune optimizer and lagrangian optimizer, making the learning of loga and 
        lagrangian terms to be an adversarial game.
        
        sprune optimizer will manage the loga parameters.

        lagrangian optimizer will manage the lagrangian terms.
        """
        if self.mode == 'tuning':
            self.sp_optimizer = None
            self.lagrangian_optimizer = None
            return
        l0_params = [{
                        "params": [p for n, p in self.named_parameters() if '_loga' in n],
                        "weight_decay": 0.0,
                        "lr": 0.1
                        }]
        self.sp_optimizer = torch.optim.AdamW(l0_params)

        lagrangian_params = [{
                    "params": [self.lambda_1, self.lambda_2],
                    "weight_decay": 0.0,
                    "lr": -0.1
                }]
        self.lagrangian_optimizer = torch.optim.AdamW(lagrangian_params)
    
    def update(self):
        r"""
        update the sprune parameters and lagrangian parameters.
        """
        if self.mode == 'training':
            sparsity = self.forward(True)
            loss = self.get_loss(sparsity)
            if torch.abs(sparsity - self.target_sparsity) < 5e-5:
                bmt.print_rank("binarize the mask and begin finetune...")
                sparsity = self.forward(False)
                for v in self.parameters():
                    v.requires_grad_(False)
                self.training = False
            return loss, sparsity
        else:
            return torch.tensor(0), torch.tensor(-1000)

    def forward(self, training: bool = True):
        for k, v in self.named_parameters():
            if "loga" not in k:
                continue
            module = k.split('_loga')[0]

            mask = sample(v) if training is True else binarize(v)
            train_mask = determinate_mask(v)
            
            for index in range(mask.size(0)):
                self.plugin.__dict__[module][index].mask = mask[index].clone().detach()
                self.plugin.__dict__[module][index].density = train_mask[index].squeeze(-1) if train_mask[index].size(-1) == 1 else train_mask[index]

        param_exp, param_all = self.plugin.get_param_exp_all()

        expected_sparsity = 1 - param_exp / param_all

        return expected_sparsity
    
    def get_loss(self, expected_sparsity):
        loss_sparsity = expected_sparsity - self.target_sparsity
        lagrangian_loss = self.lambda_1 * loss_sparsity + self.lambda_2 * (loss_sparsity ** 2)
        return lagrangian_loss

    def save(self, file_name: str):
        for k, v in self.named_parameters():
            if "loga" not in k:
                continue
            module = k.split('_loga')[0]
            mask = binarize(v, hard_binarize=self.strategy.hard_binarize, target_s=self.target_sparsity)
            
            for index in range(mask.size(0)):
                self.plugin.__dict__[module][index].mask = mask[index].clone().detach()
        self.plugin.save_masks(file_name)
