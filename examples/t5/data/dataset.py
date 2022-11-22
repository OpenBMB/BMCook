import numpy as np
import torch
import torch.utils.data as data
from . import MMapIndexedDataset


class Dataset(data.Dataset):
    def __init__(self, ctx_context : MMapIndexedDataset, ctx_target : MMapIndexedDataset, ctx_len, tar_len):
        self.ctx_context = ctx_context
        self.ctx_target = ctx_target
        self.ctx_len = ctx_len
        self.tar_len = tar_len

    def __len__(self):
        return len(self.ctx_context)
    
    def __getitem__(self, index):
        ctx_context = self.ctx_context[index]
        ctx_target = self.ctx_target[index]

        if ctx_context.shape[0] > self.ctx_len or ctx_target.shape[0] > self.tar_len:
            return None

        len_ctx_context = ctx_context.shape[0]
        len_ctx_target = ctx_target.shape[0]
        
        th_ctx_context = torch.zeros(self.ctx_len, dtype=torch.long)
        th_ctx_context[:len_ctx_context] = torch.from_numpy(ctx_context.astype(np.int))[:len_ctx_context].long()

        th_ctx_target = torch.zeros(self.tar_len, dtype=torch.long)
        th_ctx_target[:len_ctx_target-1] = torch.from_numpy(ctx_target.astype(np.int))[:len_ctx_target-1].long()

        th_target = torch.zeros(self.tar_len, dtype=torch.long) * -100
        th_target[:len_ctx_target-1] = torch.from_numpy(ctx_target.astype(np.int))[1:len_ctx_target].long()
        return {
            "ctx_context": th_ctx_context,
            "len_ctx_context": len_ctx_context,
            "ctx_target": th_ctx_target,
            "len_ctx_target": len_ctx_target-1,
            'target': th_target
        }