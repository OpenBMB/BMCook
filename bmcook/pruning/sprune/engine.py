import torch
import bmtrain as bmt
from torch.nn.parameter import Parameter
from model_center.model import BaseModel
from .func import sample_mask, binarize_mask, cdf_concrete_dist
from .plugin import SPruneStrategy, SPrunePlugin

class SPruneEngine:
    '''
    SPruneEngine is used for the computation of SPrunePlugin
    '''
    
    def __init__(self, model: BaseModel) -> None:
        super().__init__()
        assert hasattr(model, "sprune_plugin")
        self.lambda_1 = Parameter(torch.tensor(0.0, dtype=torch.float, device='cuda'))
        self.lambda_2 = Parameter(torch.tensor(0.0, dtype=torch.float, device='cuda'))

        self.training_loga = {}
        for mask, shape in model.sprune_plugin._info['shape'].items():
            self.training_loga[mask+'_loga'] = Parameter(torch.empty(shape[0], dtype=torch.float, device='cuda').normal_(10., 1e-2))
        self.plugin = model.sprune_plugin
        self.model = model
        self.target_sparsity = 0.5
        #self.target_sparsity = config.target_sparsity

        self.create_sprune_optimizer()

    def create_sprune_optimizer(self):
        l0_params = [{
                    "params": [self.lambda_1, self.lambda_2],
                    "weight_decay": 0.0,
                    "lr": 0.1
                }]
        self.sp_optimizer = torch.optim.AdamW(l0_params)

        lagrangian_params = [{
                            "params": [p for _, p in self.training_loga.items()],
                            "weight_decay": 0.0,
                            "lr": -0.1
                            }]
        self.lagrangian_optimizer = torch.optim.AdamW(lagrangian_params)
    
    def update(self, plugin: SPrunePlugin):
        self.forward(plugin, training=True)
        loss, _ = self.loss()
        loss.backward()
        self.step()
        self.zero_grad()

    def step(self):
        self.sp_optimizer.step()
        self.lagrangian_optimizer.step()
    
    def zero_grad(self):
        self.sp_optimizer.zero_grad()
        self.lagrangian_optimizer.zero_grad()

    def loss(self):
        if self.plugin._strategy.target_mode == 'sparsity':
            return self.lagrangian_loss_sparsity()
        elif self.plugin._strategy.target_mode == 'dimension':
            return self.lagrangian_loss_dimension()

    def forward(self, plugin: SPrunePlugin, training : bool = True):
        for k, v in self.training_loga.items():
            prefix = '_' + k.split('_loga')[0]
            mask = sample_mask(v) if training is True else binarize_mask(v)
            for index in range(mask.size(0)):
                #self.plugin.__dict__[prefix][index]['mask'] = mask[index]
                plugin.__dict__[prefix][index]['mask'] = mask[index]

    def lagrangian_loss_sparsity(self):
        expected_sparsity = self.plugin.get_sparsity()
        loss_sparsity = torch.tensor(expected_sparsity - self.target_sparsity)

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
