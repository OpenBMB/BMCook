import numpy as np
import torch
import torch.utils.data as data
from .indexed_dataset import MMapIndexedDataset

class Dataset(data.Dataset):
    def __init__(self, ctx : MMapIndexedDataset, dec_len):
        self.ctx = ctx
        self.dec_len = dec_len

    def __len__(self):
        return len(self.ctx)
    
    def __getitem__(self, index):
        ctx = self.ctx[index]

        if ctx.shape[0] > self.dec_len+1:
            return None

        len_ctx = ctx.shape[0]-1
        
        th_ctx = torch.zeros(self.dec_len, dtype=torch.long)
        th_ctx[:len_ctx] = torch.from_numpy(ctx.astype(np.int))[:len_ctx].long()

        th_target = torch.zeros(self.dec_len, dtype=torch.long)
        th_target[:len_ctx] = torch.from_numpy(ctx.astype(np.int))[1:len_ctx+1].long()
        return {
            "ctx": th_ctx,
            "len_ctx": len_ctx,
            'target': th_target
        }




