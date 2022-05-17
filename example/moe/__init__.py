import numpy as np
import torch
import cpm_kernels.torch as ct
import types
import os
import bmtrain as bmt

class BMMoE:
    '''
    BMMoE replaces the feed-forward modules in PLMs with MoE simulation modules.
    '''

    @staticmethod
    def get_hidden(model, config, forward_fn):
        moe_config = config.get('MoEfication')
        if not moe_config['is_moefy']:
            return forward_fn

        modules = get_modified_modules(model, moe_config['first_FFN_module'])

        update_forward(modules)

        def forward(model, dec_input, dec_length, targets, loss_func):
            with bmt.inspect.inspect_tensor() as inspector:
                outputs = forward_fn(
                    model, dec_input, dec_length, targets, loss_func)
            
            records = {}
            for record in inspector._summary:
                if 'moe_hidden' in record['name']:
                    records[record['name']] = record['tensor']
            
            return outputs + [records]
        return forward

    @staticmethod
    def moefy(model, num_expert, topk, checkpoint=None):
        '''
        Replace the feed-forward modules in PLMs with MoE modules according to the results of MoEfication from the checkpoint file.

        :param model: Model to MoEfy.
        :param num_expert: Number of experts.
        :param topk: Top-k for each expert.
        :param checkpoint: Path to load the MoEfication results.
        '''
        # after parameter initialization

        for layer_idx in range(len(model.dec_layers)):
            layer = model.dec_layers[layer_idx]

            path = os.path.join(checkpoint, 'gp_split', 'dec_layers.{}.ff.fc_in_weight.model'.format(layer_idx))

            if not os.path.exists(path):
                continue

            ff = layer._module.ff
            ff.moe = True
            ff.layer_idx = layer_idx

            ff.markers = torch.load(path).to("cuda:{}".format(torch.cuda.current_device()))

            label_file = os.path.join(checkpoint, 'gp_split', 'dec_layers.{}.ff.fc_in_weight'.format(layer_idx))
            labels = torch.load(label_file)
            cluster_num = max(labels)+1
            assert cluster_num == num_expert
            patterns = []
            for i in range(cluster_num):
                patterns.append(np.array(labels) == i)
            ff.patterns = torch.Tensor(patterns).cuda()

            ff.k = topk

def get_modified_modules(model, first_FFN_module):
    '''
    Get the modules that are modified by MoEfication.

    :param model: Model to get the modified modules.
    :param first_FFN_module: The index of the first feed-forward module.
    :return: The modules that are modified by MoEfication.
    '''
    modules = []
    for name, module in model.named_modules():
        if any([x in name for x in first_FFN_module]):
            modules.append(module)
    return modules

def update_forward(modules):
    inspect_name = "moe_hidden"
    def _forward(module_self, x):
        x = module_self.forward_old(x)
        bmt.inspect.record_tensor(x, inspect_name)
        return x
    
    for module in modules:
        module.forward_old = module.forward
        module.forward = types.MethodType(_forward, module)