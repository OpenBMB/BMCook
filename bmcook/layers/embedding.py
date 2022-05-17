import torch
import bmtrain as bmt
from cpm_kernels.torch.embedding import OpEmbedding
import cpm_kernels.torch as ct
import math

class Embedding(bmt.DistributedModule):
    def __init__(self, vocab_size : int, embedding_size : int, init_method : bmt.ParameterInitializer, dtype=torch.half):
        super().__init__()
        self.dim_model = embedding_size
        self.weight = bmt.DistributedParameter(torch.empty(vocab_size, embedding_size, dtype=dtype), init_method=init_method)

    def forward(self, ids : torch.Tensor):
        """
        Args:
            ids : (batch_size, seq_len)                         int32
        Returns:
            embedding : (batch_size, embedding_size, seq_len)   fp16
        """
        return OpEmbedding.apply(ids, self.weight)

    def proj(self, x : torch.Tensor):
        """
        Args:
            hidden : (batch_size, dim_model, seq_len)           int32
        Returns:
            logits : (batch, seq_len, vocab_output_size)        fp16
        """
        logits = ct.bmm(self.weight.unsqueeze(0), False, x, False, int8=False) / math.sqrt(self.dim_model)

        logits = ct.transpose(logits)   # eqauls to .transpose(1, 2)
        return logits
