from pathlib import Path
import os
import json
import random
import time

import torch
import bmpretrain as bmp

from distilling import BMDistill
from arguments import parse_args
from data import MMapIndexedDataset, Dataset
from models import GPT, GPTJ


def print_inspect(model, name):
    bmp.print_rank(
        bmp.inspect.format_summary(
            bmp.inspect.inspect_model(model, name)
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
    def forward(model, dec_input, dec_length, targets, loss_func, 
                output_hidden_states=False):
        outputs = model(
            dec_input, dec_length, output_hidden_states=output_hidden_states)
        logits = outputs[0]
        batch, seq_len, vocab_out_size = logits.size()

        loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))

        return (loss,) + outputs


def get_model(model_name: str, init_std: float) -> torch.nn.Module:
    if model_name == "gpt-j":
        return GPTJ(
            num_dec=14,
            dim_model=4096, num_heads=16, dim_head=256, dim_ff=16384,
            vocab_size=50400,
            init_std=init_std,
            position_bias_num_buckets=32, position_bias_max_distance=128,
            eps=1e-6, int8=False, dtype=torch.half
        )
    elif model_name == "gpt-relu":
        return GPT(
            num_dec=12,
            dim_model=768, num_heads=12, dim_head=64, dim_ff=2048,
            vocab_size=50400,
            init_std=init_std,
            position_bias_num_buckets=32, position_bias_max_distance=128,
            eps=1e-6, int8=False, dtype=torch.half,
        )
    else:
        raise ValueError("Invalid model name")



def main():
    bmp.init_distributed()

    args = parse_args()
    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    json.dump(vars(args), open(save_dir / 'train_args.json', 'w'), indent=2)

    bmp.print_rank(json.dumps(vars(args), indent=2))

    model = get_model(args.model, args.init_std)
    bmp.init_parameters(model)

    # Distillation
    if args.use_kd:
        assert args.model == "gpt-j", "Currently only support KD with GPT-J"
        teacher = GPTJ(
            num_dec=28,
            dim_model=4096, num_heads=16, dim_head=256, dim_ff=16384,
            vocab_size=50400,
            init_std=args.init_std,
            position_bias_num_buckets=32, position_bias_max_distance=128,
            eps=1e-6, int8=False, dtype=torch.half
        )
        bmp.init_parameters(teacher)
        bmp.load(teacher, "/data/home/scv0540/zzy/gpt-j/bm-gpt.pt")
        BMDistill.init_student(model, teacher)

        Trainer.forward = BMDistill.set_forward(
            model,
            teacher,
            Trainer.forward,
            output_kd_loss=True,
            temp=args.kd_temp,
            kd_loss_scale=args.kd_loss_scale,
            ce_logits=args.kd_ce_logits,
            mse_last_hidden=args.kd_mse_last_hidden,
            mse_hidden_states=args.kd_mse_hidn,
            mse_att=args.kd_mse_att,
            mse_emb=args.kd_mse_emb,
        )
        teacher.eval()

    #bmp.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmp.synchronize()

    # data
    batch_size = 8
    dec_len = 512

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = bmp.optim.AdamOptimizer(model.parameters(), scale=2**20)
    lr_scheduler = bmp.lr_scheduler.Noam(optimizer, start_lr=args.start_lr, warmup_iter=500, end_iter=100000)

    bmp.synchronize()
    average_time = 0
    average_time_shift = 0.9

    dataset = Dataset(
        MMapIndexedDataset("/data/home/scv0540/zzy/openwebtxt/openwebtext_text_document"),
        dec_len
    )

    sparsity_log_interval = 100

    for _ in range(10):
        for iteration, data in enumerate(Trainer.batch_iter_shuf(dataset, batch_size, bmp.rank(), bmp.world_size())):
            st = time.time()
            optimizer.zero_grad()

            dec_input = data["ctx"].int()
            dec_length = data["len_ctx"].int()
            dec_mask = torch.arange(dec_len)[None, :].repeat(batch_size, 1) < dec_length[:, None]
            targets = torch.where(dec_mask, data["target"].long(), torch.scalar_tensor(-100, dtype=torch.long))

            targets = targets.cuda()
            dec_input = dec_input.cuda()
            dec_length = dec_length.cuda()

            if args.use_kd:
                loss, _, kd_loss = Trainer.forward(
                    model, dec_input, dec_length, targets, loss_func)
                global_kd_loss = bmp.sum_loss(kd_loss).item()
            else:
                loss, _ = Trainer.forward(model, dec_input, dec_length, targets, loss_func)

            global_loss = bmp.sum_loss(loss).item()

            loss = optimizer.loss_scale(loss)
            loss.backward()

            if iteration % 1000 == 0:
                print_inspect(model, "*")
            # bmp.print_rank(bmp.inspect.format_summary(inspector.get_summary()))
            bmp.optim_step(optimizer, lr_scheduler)

            if iteration % args.log_interval == 0:
                iteration_time = time.time() - st
                average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time
                if args.use_kd:
                    bmp.print_rank(
                        "| Iter: {:6d} | loss: {:.4f} | kd_loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f}".format(
                            iteration,
                            global_loss,
                            global_kd_loss,
                            lr_scheduler.current_lr,
                            int(optimizer.scale),
                            average_time / (1 - pow(average_time_shift, iteration + 1))
                        )
                    )
                else:
                    bmp.print_rank(
                        "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f}".format(
                            iteration,
                            global_loss,
                            lr_scheduler.current_lr, 
                            int(optimizer.scale), 
                            average_time / (1 - pow(average_time_shift, iteration + 1))
                        )
                    ) 
            if iteration % args.save_interval == 0: 
                ckpt_file = Path(args.save_dir, 'checkpoints', f'ckpt-{iteration}.pt')
                bmp.save(model, ckpt_file) 
            if args.model == "gpt-relu":
                if (iteration + 1) % sparsity_log_interval == 0:
                    # sparsity
                    sparsity = model.get_sparsity()
                    mean_sparsity = sum(sparsity) / len(sparsity)
                    bmp.print_rank(f"sparsity: {sparsity}, mean: {mean_sparsity}")
                    model.reset_sparsity()

    ckpt_file = Path(args.save_dir, 'checkpoint.pt') 
    bmp.save(model, ckpt_file) 


if __name__ == '__main__': 
    main()
