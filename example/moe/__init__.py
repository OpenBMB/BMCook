import sklearn
import numpy as np
import torch
import cpm_kernels.torch as ct
import types
import os

class BMMoE:
    '''
    BMMoE replaces the feed-forward modules in PLMs with MoE modules.
    '''

    @staticmethod
    def moefy(model, num_expert, topk, checkpoint=None):
        '''
        Replace the feed-forward modules in PLMs with MoE modules according to the results of MoEfication from the checkpoint file.

        :param model: Model to MoEfy.
        :param num_expert: Number of experts.
        :param topk: Top-k for each expert.
        :param checkpoint: Path to load the MoEfication results.
        :return: MoEfied model.
        '''
        # after parameter initialization

        for i, layer in enumerate(model.dec_layers):
            ff = layer._module.ff
            ff.moe = True

            ff.markers = torch.load(os.path.join(checkpoint, 'gp_split', 'layer_{}_input_compl'.format(i)))

            label_file = os.path.join(checkpoint, 'gp_split', 'layer_{}'.format(i))
            labels = torch.load(label_file)
            cluster_num = max(labels)+1
            assert cluster_num == num_expert
            patterns = []
            for i in range(cluster_num):
                patterns.append(np.array(labels) == i)
            ff.patterns = torch.Tensor(patterns).cuda()

            ff.k = topk
