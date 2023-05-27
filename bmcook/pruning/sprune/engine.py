import torch
import bmtrain as bmt
from typing import Dict, Optional
from torch.nn.parameter import Parameter
from .func import determinate_mask, sample, binarize, from_sparsity
from .plugin import SPrunePlugin, UnitList
from .utils import inspect_checkpoint_block


class SPruneStrategy:
    def __init__(self, config: Dict) -> None:
        self.criterion = config['criterion']
        assert self.criterion == 'l0', "BMCook sprune do not support other criterions besides l0 yet."
        self.mask_path = config['mask_path']
        self.training_mask = config['training_mask']
        self.is_training = config['is_training']
        self.target_mode = config['target_mode']
        self.target_sparsity = config['target_sparsity']
        self.start_sparsity = config['start_sparsity']
        self.hard_binarize = config['hard_binarize']
        self.tuning = config['tuning']
        if self.is_training is False:
            self.iterative = self.tuning['iterative']
            self.interval = self.tuning['interval']
            self.ratio = self.tuning['ratio']


class SPruneEngine(bmt.DistributedModule):
    r"""
    SPruneEngine is used for the mask computation and update of SPrunePlugin.

    The engine design is based on L0 regularization method and a lagrangian term. For L0 regularization details, see paper
        "Learning Sparse Neural Networks through L_0 Regularization" <https://openreview.net/forum?id=H1Y8hhg0b>.
        For lagrangian term in PLM structure pruning, see paper "Structured Pruning of Large Language Models" 
        <https://arxiv.org/abs/1910.04732>.
    """
    def __init__(self, strategy: SPruneStrategy, plugin: SPrunePlugin, saved_path: str) -> None:
        r"""Init the SpruneEngine from a SPrunePlugin. It will initilize all the :class:`torch.nn.Parameter`
        used for learning the sprune mask, and create the optimizer for l0 regularization.

        Args:
            config: `(Dict)`, the sprune config.
            plugin: `(SPrunePlugin)`, the SPrunePlugin.
        """
        super().__init__()
        self.strategy = strategy
        self.target_sparsity = self.strategy.target_sparsity
        self.is_training = self.strategy.is_training
        self.plugin = plugin
        self.loga_param = None
        
        if self.is_training:
            #TODO from sparsity to loga
            start_loga = from_sparsity(self.strategy.start_sparsity)

            self.lambda_1 = Parameter(torch.tensor(0., dtype=torch.float, device='cuda'))
            self.lambda_2 = Parameter(torch.tensor(0., dtype=torch.float, device='cuda'))

            for name, unitlist in self.plugin.named_lists():
                for i, unit in enumerate(unitlist):
                    setattr(self, f"{name}-{i}-loga", Parameter(torch.ones_like(unit.mask, dtype=torch.float).normal_(start_loga, 0.01)))
            self.loga_param = [(n, p) for n, p in self.named_parameters() if '-loga' in n]
            self.lambda_param = [self.lambda_1, self.lambda_2]
            self.create_sprune_optimizer()

        else:
            self.target_mask = torch.load(saved_path)
            if not self.strategy.iterative:
                for k, _ in self.plugin.named_lists():
                    if k in self.target_mask:
                        cur_unitlist = getattr(self.plugin, k)
                        cur_unitlist.load_mask(self.target_mask[k])
            else:
                self.interval = self.strategy.interval
                self.iter_pruner = self.base_iterative_prune()

    def create_sprune_optimizer(self):
        r"""Create the sprune optimizer and lagrangian optimizer, making the learning of loga and 
        lagrangian terms to be an adversarial game.
        
        sprune optimizer will manage the loga parameters.

        lagrangian optimizer will manage the lagrangian terms.
        """
        l0_params = [{
                        "params": [p for (_, p) in self.loga_param],
                        "weight_decay": 0.0,
                        "lr": 0.1
                        }]
        self.sp_optimizer = torch.optim.AdamW(l0_params) if self.is_training else None

        lagrangian_params = [{
                    "params": self.lambda_param,
                    "weight_decay": 0.0,
                    "lr": -0.1
                }]
        self.lagrangian_optimizer = torch.optim.AdamW(lagrangian_params) if self.is_training else None
    
    def update(self):
        r"""
        update the sprune parameters and lagrangian parameters.
        """
        if self.is_training:
            sparsity = self.forward(True)
            loss = self.get_loss(sparsity)
            if torch.abs(sparsity - self.target_sparsity) < 5e-5:
                bmt.print_rank("binarize the mask and begin finetune...")
                sparsity = self.forward(False)
                for v in self.parameters():
                    v.requires_grad_(False)
                self.is_training = False
            return loss, sparsity
        else:
            return torch.tensor(0), torch.tensor(-1000)

    def forward(self, training: bool = True):
        for (k, v) in self.loga_param:
            list_name, unit_index, _ = k.split('-')
            mask = sample(v) if training is True else binarize(v)
            train_mask = determinate_mask(v)
            unit_list = getattr(self.plugin, list_name)
            unit_list[int(unit_index)].mask = mask.half().clone().detach()
            unit_list[int(unit_index)].density = train_mask.sum()

        param_exp, param_all = 0, 0
        for transformer_unit in self.plugin:
            param_exp += transformer_unit.get_param_exp()
            param_all += transformer_unit.get_param_all()

        expected_sparsity = 1 - param_exp / param_all

        return expected_sparsity
    
    def get_loss(self, expected_sparsity):
        loss_sparsity = expected_sparsity - self.target_sparsity
        sp_loss = self.lambda_1 * loss_sparsity + self.lambda_2 * (loss_sparsity ** 2)
        return sp_loss

    def save_masks(self, file_name: str):
        if self.loga_param is not None:
            for (k, v) in self.loga_param:
                list_name, unit_index, _ = k.split('-')
                mask = determinate_mask(v)
                unit_list = getattr(self.plugin, list_name)
                unit_list[int(unit_index)].mask = mask.half().clone().detach()
        
        for _, unit_list in self.plugin.named_lists():
            unit_list.binarize_mask(hard_binarize=self.strategy.hard_binarize, target_s=self.strategy.target_sparsity)
        
        self.plugin.save_masks(file_name)
