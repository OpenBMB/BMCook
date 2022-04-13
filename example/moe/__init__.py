import numpy as np
import torch
import cpm_kernels.torch as ct
import types
import os

class BMMoE:
    '''
    BMMoE replaces the feed-forward modules in PLMs with MoE simulation modules.
    '''

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
