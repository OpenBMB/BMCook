import torch
import cpm_kernels.torch as ct
import bmpretrain as bmp
import math
import numpy as np

from .average import Average

class FeedForward(bmp.DistributedModule):
    def __init__(self, 
            dim_model : int,
            dim_ff : int,
            init_method : bmp.ParameterInitializer,
            int8=True,
            dtype=torch.half
        ):
        super().__init__()
        self.w_0 = bmp.DistributedParameter(
            torch.empty(dim_ff, dim_model, dtype=dtype), init_method=init_method)
        self.w_out = bmp.DistributedParameter(
            torch.empty(dim_model, dim_ff, dtype=dtype), init_method=init_method)

        self.relu = torch.nn.ReLU()

        self.int8 = int8
        self.dim_model = dim_model
        self.dim_ff = dim_ff

        self.sparsity = Average()
        self.relu_distr = Average()
        self.record_relu_distr = False
    
    def forward(self, x):
        """
        Args:
            x : (batch, hidden_size, seq_len)       fp16
        Returns:
            out : (batch, hidden_size, seq_len)     fp16
        """
        # (1#batch, dim_ff, dim_model) @ (batch, dim_model, seq_len) = (batch, dim_ff, seq_len)

        # bmp.inspect.record_tensor(x, "ff_x")

        w_0 = self.w_0
        w_out = self.w_out

        x = self.relu(
            ct.bmm(w_0.unsqueeze(0), False, x, False, int8=self.int8) #/ math.sqrt(self.dim_model)
        )
        zero_prop = (x == 0).sum().item() / x.numel()
        self.sparsity.add(zero_prop)

        # Record sparsity
        nonzero_prop = torch.count_nonzero(x) / x.numel()
        self.sparsity.add(1 - nonzero_prop.item())

        # Record distribution of output of ReLU
        if self.record_relu_distr:
            x_values = np.sort(x.cpu().detach().numpy().flatten())
            self.relu_distr.add(x_values)

        # (1#batch, dim_model, dim_ff) @ (batch, dim_ff, seq_len) = (batch, dim_model, seq_len)
        x = ct.bmm(w_out.unsqueeze(0), False, x, False, int8=self.int8) #/ math.sqrt(self.dim_ff)
        return x

def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class GPTJFF(bmp.DistributedModule):
    def __init__(self, dim_model : int, dim_ff : int, init_method : bmp.ParameterInitializer, act_func='gelu' , int8=True, dtype=torch.half):
        super().__init__()
        self.fc_in_weight = bmp.DistributedParameter(torch.empty(dim_ff, dim_model, dtype=dtype), init_method=init_method)
        self.fc_in_bias = bmp.DistributedParameter(torch.zeros(1, dim_ff, 1, dtype=dtype))

        self.fc_out_weight = bmp.DistributedParameter(torch.empty(dim_model, dim_ff, dtype=dtype), init_method=init_method)
        self.fc_out_bias = bmp.DistributedParameter(torch.zeros(1, dim_model, 1, dtype=dtype))

        self.act_func = act_func
        if act_func == 'gelu':
            self.relu = gelu_new
        else:
            self.relu = torch.nn.ReLU()

        self.int8 = int8
        self.dim_model = dim_model
        self.dim_ff = dim_ff

        self.moe = None

    def forward(self, x):
        """
        Args:
            x : (batch, hidden_size, seq_len)       fp16
        Returns:
            out : (batch, hidden_size, seq_len)     fp16
        """
        # (1#batch, dim_ff, dim_model) @ (batch, dim_model, seq_len) = (batch, dim_ff, seq_len)

        bsz, hidden_size, seq_len = x.shape

        bmp.inspect.record_tensor(x, "ff_x")

        w_0 = self.fc_in_weight
        w_out = self.fc_out_weight

        if self.moe is not None:
            xx = x.float().transpose(1,2).view(-1, hidden_size)
            xx = xx / torch.norm(xx, dim=-1).unsqueeze(-1)

            score = self.markers(xx)
            labels = torch.topk(score, k=self.k, dim=-1)[1].view(bsz, seq_len, self.k)
            cur_mask = torch.nn.functional.embedding(labels, self.patterns).sum(-2).transpose(1,2)

        if self.act_func == 'gelu':
            x = self.relu(
                ct.bmm(w_0.unsqueeze(0), False, x, False, int8=self.int8) + self.fc_in_bias
            )
        else:
            x = self.relu(
                ct.bmm(w_0.unsqueeze(0), False, x, False, int8=self.int8)
            )
        
        if self.moe:
            x[cur_mask == False] = 0

        # (1#batch, dim_model, dim_ff) @ (batch, dim_ff, seq_len) = (batch, dim_model, seq_len)
        
        x = ct.bmm(w_out.unsqueeze(0), False, x, False, int8=self.int8) + self.fc_out_bias #/ math.sqrt(self.dim_ff)
        return x

