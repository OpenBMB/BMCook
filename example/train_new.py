import json
import torch
import random
import bmtrain as bmt
from transformers import GPT2ForTokenClassification
from moe import BMMoE
from quant import BMQuant
import layers
from tqdm import tqdm
import time
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

from data import MMapIndexedDataset, Dataset
import numpy as np
import pickle as pkl
from pruning import BMPrune, m4n2_2d_greedy
from model_center.model import GPT2Config, GPT2
from distilling import BMDistill
from arguments import parse_args
from pathlib import Path
import os
import json

import os
from arguments import parse_args
from pathlib import Path

from utils.config import ConfigParser

def print_inspect(model, name):
    bmt.print_rank(
        bmt.inspect.format_summary(
            bmt.inspect.inspect_model(model, name)
        )
    )

class Trainer:
    @staticmethod
    def batch_iter_shuf(dataset : Dataset, batch_size, rank, world_size):
        st = 0
        end = len(dataset)

        local_size = len(dataset) // world_size
        idx = list(range(local_size))
        random.shuffle(idx)

        batch = []
        while st < local_size:
            it = dataset[idx[st]*world_size + rank]
            if it is not None:
                batch.append( it )
            st += 1
            if len(batch) == batch_size:
                yield {
                    "ctx": torch.stack([it["ctx"] for it in batch]),
                    "len_ctx": torch.LongTensor([it["len_ctx"] for it in batch]),
                    'target': torch.stack([it["target"] for it in batch]),
                }
                batch = []

    @staticmethod
    def batch_iter(dataset : Dataset, batch_size, rank, world_size):
        st = 0
        end = len(dataset)
        batch = []
        while st < end:
            it = dataset[st + rank]
            if it is not None:
                batch.append( it )
            st += world_size
            if len(batch) == batch_size:
                yield {
                    "ctx": torch.stack([it["ctx"] for it in batch]),
                    "len_ctx": torch.LongTensor([it["len_ctx"] for it in batch]),
                    'target': torch.stack([it["target"] for it in batch]),
                }
                batch = []

    @staticmethod
    def forward(model, dec_input, dec_length, targets, loss_func):
        outputs = model(
            dec_input, dec_length, return_logits=True)
        logits = outputs
        batch, seq_len, vocab_out_size = logits.size()

        loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))

        return [loss, logits]


def main():
    bmt.init_distributed(seed=1)

    # args = parse_args()
    # save_dir = Path(args.save_dir)
    # ckpt_dir = save_dir / 'checkpoints'
    # os.makedirs(ckpt_dir, exist_ok=True)
    # json.dump(vars(args), open(save_dir / 'train_args.json', 'w'), indent=2)

    # model = get_model(args.model, args.init_std)
    # bmt.init_parameters(model)

    gpt_config = GPT2Config.from_pretrained("gpt2-base")
    # gpt_config.dropout_p = 0

    gpt = GPT2.from_pretrained("gpt2-base", config=gpt_config)
    teacher = GPT2.from_pretrained("gpt2-base", config=gpt_config)

    config = ConfigParser('/home/zhangzhengyan/BMCook/example/configs/test.json')
    Trainer.forward = BMDistill.set_forward(gpt, teacher, Trainer.forward, config)

    #print_inspect(model, "*")
    #bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()

    batch_size = 2
    seq_len = 32

    for i in range(bmt.world_size()):
        sent = torch.randint(0, 10240, (batch_size, seq_len + 1))
        enc_length = torch.randint(16, seq_len, (batch_size,)).long().cuda()
        enc_input = sent[:, :-1].long().cuda()
        targets = sent[:, 1:].long().cuda()
        mask = torch.arange(seq_len).long().cuda()[None, :] < enc_length[:, None]
        targets = torch.where(
            mask,
            targets,
            torch.full_like(targets, -100, dtype=torch.long)
        )

        if i == bmt.rank():
            break

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = bmt.optim.AdamOptimizer(gpt.parameters(), scale=2**20)
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=1e-4, warmup_iter=2000, end_iter=100000)

    # for pruning
    BMPrune.compute_mask(gpt, config)
    BMPrune.set_optim_for_pruning(optimizer)

    # for quantization
    BMQuant.quantize(gpt, config)

    # if args.moe:
    #     from moe import BMMoE
    #     BMMoE.moefy(model, args.num_expert, args.topk, checkpoint=args.moe_ckpt)

    Trainer.forward = BMMoE.get_hidden(gpt, config, Trainer.forward)

    bmt.synchronize()

    # dataset = Dataset(
    #     MMapIndexedDataset("/data/home/scv0540/zzy/openwebtxt/openwebtext_text_document"),
    #     dec_len
    # )

    # if args.eval:
    #     eval_losses = []

    # if args.save_hidden:
    #     os.makedirs(save_dir / 'hiddens', exist_ok=True)
    #     model.eval()

    #     for iteration, data in enumerate(Trainer.batch_iter(dataset, batch_size, bmt.rank(), bmt.world_size())):

    #         if iteration == 100:
    #             break

    #         with bmt.inspect.inspect_tensor() as inspector:

    #             dec_input = data["ctx"].int()
    #             dec_length = data["len_ctx"].int()
    #             dec_mask = torch.arange(dec_len)[None, :].repeat(batch_size, 1) < dec_length[:, None]
    #             targets = torch.where(dec_mask, data["target"].long(), torch.scalar_tensor(-100, dtype=torch.long))

    #             targets = targets.cuda()
    #             dec_input = dec_input.cuda()
    #             dec_length = dec_length.cuda()
                
    #             with torch.no_grad():
    #                 loss, logits, _, _ = Trainer.forward(model, dec_input, dec_length, targets, loss_func)
            
    #         tensors = [x['tensor'] for x in inspector._summary if 'ff_x' in x['name']]
               
    #         torch.save(tensors, save_dir / 'hiddens' / '{}_{}'.format(iteration, bmt.rank()) )
    #         bmt.print_rank("Iteration:", iteration)
    #     exit()

    for iteration in range(1000):
        # gpt.eval()
        teacher.eval()
        optimizer.zero_grad()

        outputs = Trainer.forward(
            gpt, enc_input, enc_length, targets, loss_func)

        if config.get('MoEfication')['is_moefy']:
            torch.save(outputs[-1], 'hiddens/' + '{}_{}'.format(iteration, bmt.rank()))

        loss = outputs[0]
        global_loss = bmt.sum_loss(loss).item()
        loss = optimizer.loss_scale(loss)

        loss.backward()
        bmt.optim_step(optimizer, lr_scheduler)

        bmt.print_rank("Iteration:", iteration, "Loss:", global_loss)
        


if __name__ == '__main__': 
    main()
