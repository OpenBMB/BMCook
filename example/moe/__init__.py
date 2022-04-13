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

        To use this method, you need to implement router operation in FFNs as follows:

        ```python
        if self.moe is not None:
            with torch.no_grad():
                xx_ = input.float().transpose(1,2).reshape(-1, hidden_size)
                xx = xx_ / torch.norm(xx_, dim=-1).unsqueeze(-1)

                score = self.markers(xx)
                labels = torch.topk(score, k=self.k, dim=-1)[1].reshape(bsz, seq_len, self.k)
                cur_mask = torch.nn.functional.embedding(labels, self.patterns).sum(-2).transpose(1,2).detach()
        ```

        ```python
        if self.moe is not None:
            inter_hidden[cur_mask == False] = 0
        ```

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
