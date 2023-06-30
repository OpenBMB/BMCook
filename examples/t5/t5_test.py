import os
import json
import torch
import random
import time
import bmtrain as bmt
from data import MMapIndexedDataset, Dataset
from bmcook import CookTrainer
from bmcook.utils.config import ConfigParser
from bmcook.utils.arguments import parse_args
from pathlib import Path

def print_inspect(model, name):
    bmt.print_rank(
        bmt.inspect.format_summary(
            bmt.inspect.inspect_model(model, name)
        )
    )

class Dataloader:
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
                    "ctx_context": torch.stack([it["ctx_context"] for it in batch]),
                    "len_ctx_context": torch.LongTensor([it["len_ctx_context"] for it in batch]),
                    "ctx_target": torch.stack([it["ctx_target"] for it in batch]),
                    "len_ctx_target": torch.LongTensor([it["len_ctx_target"] for it in batch]),
                    'target': torch.stack([it["target"] for it in batch]),
                }
                batch = []

from model_center.model import GPT2Config, GPT2, T5Config, T5

config_map = {
    'gpt2-base': GPT2Config,
    't5-3b': T5Config,
    't5-large': T5Config,
    't5-base': T5Config,
}

model_map = {
    'gpt2-base': GPT2,
    't5-3b': T5,
    't5-large': T5,
    't5-base': T5,
}

def main():
    bmt.init_distributed()

    args = parse_args()
    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    json.dump(vars(args), open(save_dir / 'train_args.json', 'w'), indent=2)

    model_config = config_map[args.model].from_pretrained(args.model)
    model_config.scale = True
    model = model_map[args.model].from_pretrained(args.model, config=model_config)
    
    # teacher model has the same config as the student model
    teacher = model_map[args.model].from_pretrained(args.model, config=model_config)

    bmt.synchronize()

    # data
    batch_size = 4
    ctx_len = 512
    tar_len = 256

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = bmt.optim.AdamOptimizer(model.parameters())
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=args.start_lr, warmup_iter=2000, end_iter=100000)
    optim_manager = bmt.optim.OptimManager(loss_scale=2**20)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    config = ConfigParser(args.cook_config)
    CookTrainer.set_compression(config, model, optimizer, teacher)    
    
    average_time = 0
    average_time_shift = 0.9

    dataset = Dataset(
        MMapIndexedDataset(args.data_path+'_context'),
        MMapIndexedDataset(args.data_path+'_target'),
        ctx_len,
        tar_len
    )


    if config.get('MoEfication')['is_moefy']:
        os.makedirs(save_dir / 'hiddens', exist_ok=True)
        model.eval()

        hiddens_dir = save_dir / 'hiddens'
        os.makedirs(hiddens_dir, exist_ok=True)

        for iteration, data in enumerate(Dataloader.batch_iter(dataset, batch_size, bmt.rank(), bmt.world_size())):

            if iteration == 100:
                break

            enc_input = data["ctx_context"].int()
            enc_length = data["len_ctx_context"].int()

            enc_input = enc_input.cuda()
            enc_length = enc_length.cuda()

            dec_input = data["ctx_target"].int()
            dec_length = data["len_ctx_target"].int()
            dec_mask = torch.arange(tar_len)[None, :].repeat(batch_size, 1) < dec_length[:, None]
            targets = torch.where(dec_mask, data["target"].long(), torch.scalar_tensor(-100, dtype=torch.long))

            targets = targets.cuda()
            dec_input = dec_input.cuda()
            dec_length = dec_length.cuda()
            
            with torch.no_grad():
                outputs = CookTrainer.forward(model, loss_func, targets, enc_input, enc_length, dec_input, dec_length)
            
            torch.save(outputs.moe_records, hiddens_dir / '{}_{}'.format(iteration, bmt.rank()))
               
            bmt.print_rank("Iteration:", iteration)
        exit()

    do_distill = True
    distill_config = config.get('distillation')
    if distill_config['ce_scale'] + distill_config['mse_hidn_scale'] + distill_config['mse_att_scale'] == 0:
        do_distill = False

    model.train()
    teacher.eval()

    for epoch in range(3):
        
        for iteration, data in enumerate(Dataloader.batch_iter(dataset, batch_size, bmt.rank(), bmt.world_size())):

            st = time.time()
            optim_manager.zero_grad()

            enc_input = data["ctx_context"].int()
            enc_length = data["len_ctx_context"].int()

            enc_input = enc_input.cuda()
            enc_length = enc_length.cuda()

            dec_input = data["ctx_target"].int()
            dec_length = data["len_ctx_target"].int()
            dec_mask = torch.arange(tar_len)[None, :].repeat(batch_size, 1) < dec_length[:, None]
            targets = torch.where(dec_mask, data["target"].long(), torch.scalar_tensor(-100, dtype=torch.long))

            targets = targets.cuda()
            dec_input = dec_input.cuda()
            dec_length = dec_length.cuda()

            outputs = CookTrainer.forward(model, loss_func, targets, enc_input, enc_length, dec_input, dec_length)

            loss = outputs.loss
            
            global_loss = bmt.sum_loss(loss).item()
            optim_manager.backward(loss)

            if do_distill:
                distill_loss = bmt.sum_loss(outputs.d_loss).item()
            else:
                distill_loss = 0
            
            optim_manager.step()

            if iteration % args.log_interval == 0:
                iteration_time = time.time() - st
                average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time
                bmt.print_rank(
                    "| Iter: {:6d} | loss: {:.4f} | kd_loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f}".format(
                        iteration,
                        global_loss-distill_loss,
                        distill_loss,
                        lr_scheduler.current_lr,
                        int(optim_manager.loss_scale),
                        average_time / (1 - pow(average_time_shift, iteration + 1))
                    )
                )
            
            if iteration % args.save_interval == 0: 
                ckpt_file = Path(args.save_dir, 'checkpoints', f'ckpt-{iteration}.pt')
                bmt.save(model, ckpt_file) 

    ckpt_file = Path(args.save_dir, 'checkpoint.pt') 
    bmt.save(model, ckpt_file) 
    bmt.synchronize()


if __name__ == '__main__': 
    main()
