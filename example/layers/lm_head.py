import torch
import bmtrain as bmt
from cpm_kernels.torch.embedding import OpEmbedding
import cpm_kernels.torch as ct
import math

class LMHead(bmp.DistributedModule):
    def __init__(self, vocab_size : int, embedding_size : int, init_method : bmp.ParameterInitializer, dtype=torch.half):
        super().__init__()
        self.dim_model = embedding_size
        self.weight = bmp.DistributedParameter(torch.empty(vocab_size, embedding_size, dtype=dtype), init_method=init_method)
        self.bias = bmp.DistributedParameter(torch.empty(1, vocab_size, 1, dtype=dtype), init_method=bmp.ParameterInitializer(torch.nn.init.zeros_))

    def forward(self, x : torch.Tensor):
        """
        Args:
            hidden : (batch_size, dim_model, seq_len)           int32
        Returns:
            logits : (batch, seq_len, vocab_output_size)        fp16
        """
        logits = ct.bmm(self.weight.unsqueeze(0), False, x, False, int8=False) + self.bias # / math.sqrt(self.dim_model)

        logits = ct.transpose(logits)  # eqauls to .transpose(1, 2)
        return logits
