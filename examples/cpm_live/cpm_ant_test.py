import torch
import random
import bmtrain as bmt
import numpy as np
from bmcook.utils.config import ConfigParser
from bmcook import CookTrainer

import sys, os
sys.path.insert(0, os.getcwd())
from cpm_live.arguments import parse_args
from cpm_live.models import CPMAntConfig, CPMAnt
from cpm_live.tokenizers import CPMAntTokenizer
from cpm_live.data import DistributedDataset

class CPMAntPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, ctx, max_length=1024, prompt_length=32, tokenizer=None):
        self.ctx = ctx
        self.max_length = max_length + prompt_length
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ctx)

    @property
    def dataset(self):
        return self.ctx

    def __get_item_data(self, raw_data):

        global_task = raw_data[0]
        n_segment = raw_data[1]
        len_info = n_segment * 3 + 2
        segment_len = raw_data[2:len_info:3]
        segment_type = raw_data[3:len_info:3]
        segment_task = raw_data[4:len_info:3]
        ctx = raw_data[len_info:]

        if ctx.shape[0] > self.max_length - self.prompt_length:
            return None, None, None, None, None, None, None
        len_ctx = min(ctx.shape[0], self.max_length - self.prompt_length)

        context_inp = np.full(len_ctx, True)
        position_inp = np.arange(len_ctx, dtype=np.int64)
        segment_inp = np.full(len_ctx, 0, dtype=np.int64)
        task_inp = np.full(len_ctx, 0, dtype=np.int64)
        tgt = np.full(len_ctx, -100, dtype=np.int64)

        # for each segment
        segment_begin = 0
        for i in range(n_segment):
            segment_end = segment_begin + segment_len[i]
            task = segment_task[i]
            # generate target
            if task == 0:
                num_mask = random.randint(1, segment_len[i] - 1)
                mask_idx = (
                    np.random.choice(segment_len[i] - 1, num_mask, replace=False) + segment_begin
                )
                context_inp[mask_idx + 1] = False
                assert segment_type[i] == 1
            elif task == 1:
                num_mask = random.randint(1, segment_len[i] - 1)
                context_inp[segment_end - num_mask : segment_end] = False
                assert segment_type[i] == 2
            elif task == 3:
                if segment_type[i] == 2:
                    context_inp[1:] = False
            elif task == 4:
                if segment_type[i] == 3:
                    context_inp[1:] = False
            task_inp[segment_begin:segment_end] = task
            segment_inp[segment_begin:segment_end] = segment_type[i]
            tgt[segment_begin : segment_end - 1] = np.where(
                context_inp[segment_begin + 1 : segment_end],
                -100,
                ctx[segment_begin + 1 : segment_end],
            )
            segment_begin = segment_end
        # prepend prompt segment
        context_inp = np.concatenate((np.full(self.prompt_length, True), context_inp))
        position_inp = np.concatenate(
            (
                np.arange(self.prompt_length, dtype=np.int64),
                position_inp + self.prompt_length,
            )
        )
        segment_inp = np.concatenate((np.full(self.prompt_length, 0, dtype=np.int64), segment_inp))
        task_inp = np.concatenate((np.full(self.prompt_length, 0, dtype=np.int64), task_inp))
        tgt = np.concatenate((np.full(self.prompt_length, -100, dtype=np.int64), tgt))
        inp = np.concatenate(
            (
                np.arange(self.prompt_length, dtype=np.int64) + self.prompt_length * global_task,
                ctx,
            )
        )
        return inp, tgt, inp.shape[0], context_inp, position_inp, segment_inp, task_inp

    def __iter__(self):
        while True:
            ctx = self.ctx.read()
            (
                th_ctx,
                th_tgt,
                len_ctx,
                context_ctx,
                position_ctx,
                segment_ctx,
                task_ctx,
            ) = self.__get_item_data(ctx)
            yield th_ctx, th_tgt, len_ctx, context_ctx, position_ctx, segment_ctx, task_ctx

class BatchPacker:
    def __init__(self, dataset, max_length, batch_size):
        self.dataset = dataset
        self.max_length = max_length
        self.batch_size = batch_size

    def __iter__(self):
        ctx = []
        tgt = []
        context = []
        position = []
        segment = []
        span = []
        task_info = []

        for data in self.dataset:
            (
                ctx_data,
                tgt_data,
                _len,
                context_data,
                position_data,
                segment_data,
                task_data,
            ) = data
            if ctx_data is None:
                continue
            assert _len <= self.max_length

            ctx_data = ctx_data.astype("int64")
            tgt_data = tgt_data.astype("int64")

            for index in range(len(ctx)):
                if span[index][-1] + _len < self.max_length:
                    ctx[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        ctx_data
                    )[:_len].long()
                    tgt[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        tgt_data
                    )[:_len].long()
                    context[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        context_data
                    )[:_len].bool()
                    position[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        position_data
                    )[:_len].long()
                    segment[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        segment_data
                    )[:_len].long()
                    task_info[index][span[index][-1] : span[index][-1] + _len] = torch.from_numpy(
                        task_data
                    )[:_len].long()
                    span[index].append(span[index][-1] + _len)
                    break
            else:

                _ctx = torch.zeros((self.max_length,), dtype=torch.long)
                _ctx[:_len] = torch.from_numpy(ctx_data)[:_len].long()
                _tgt = torch.full((self.max_length,), -100, dtype=torch.long)
                _tgt[:_len] = torch.from_numpy(tgt_data)[:_len].long()
                _context = torch.full((self.max_length,), False, dtype=torch.bool)
                _context[:_len] = torch.from_numpy(context_data)[:_len].bool()
                _position = torch.full((self.max_length,), False, dtype=torch.long)
                _position[:_len] = torch.from_numpy(position_data)[:_len].long()
                _segment = torch.full((self.max_length,), False, dtype=torch.long)
                _segment[:_len] = torch.from_numpy(segment_data)[:_len].long()
                _task_info = torch.full((self.max_length,), -1, dtype=torch.long)
                _task_info[:_len] = torch.from_numpy(task_data)[:_len].long()
                ctx.append(_ctx)
                tgt.append(_tgt)
                context.append(_context)
                position.append(_position)
                segment.append(_segment)
                task_info.append(_task_info)
                span.append([_len])

            if len(ctx) > self.batch_size:
                _span = torch.zeros((self.batch_size, self.max_length + 1), dtype=torch.long)
                for bindex in range(self.batch_size):
                    for sindex in span[bindex]:
                        _span[bindex][sindex] = 1

                yield {
                    "ctx": torch.stack(ctx[: self.batch_size]),
                    "tgt": torch.stack(tgt[: self.batch_size]),
                    "context": torch.stack(context[: self.batch_size]),
                    "segment": torch.stack(segment[: self.batch_size]),
                    "position": torch.stack(position[: self.batch_size]),
                    "span": torch.cumsum(_span, dim=-1)[:, :-1],
                    "len_ctx": torch.LongTensor([it[-1] for it in span[: self.batch_size]]),
                    "task": torch.stack(task_info[: self.batch_size]),
                }

                ctx = ctx[self.batch_size :]
                tgt = tgt[self.batch_size :]
                context = context[self.batch_size :]
                segment = segment[self.batch_size :]
                position = position[self.batch_size :]
                span = span[self.batch_size :]
                task_info = task_info[self.batch_size :]

def main():

    bmt.init_distributed()

    args = parse_args()
    batch_size = 4
    max_length = 512
    prompt_length = 32

    config = CPMAntConfig.from_json_file(args.model_config)
    model = CPMAnt(config)
    bmt.load(model, args.load)

    teacher_config = CPMAntConfig.from_json_file(args.teacher_config)
    teacher = CPMAnt(teacher_config)
    bmt.load(teacher, args.load_teacher)

    config = ConfigParser(args.cook_config)
    do_distill = True
    distill_config = config.get('distillation')
    if distill_config['ce_scale'] + distill_config['mse_hidn_scale'] + distill_config['mse_att_scale'] == 0:
        do_distill = False

    tokenizer = CPMAntTokenizer('vocabs/ant.txt')
    dataset = CPMAntPretrainDataset(
        DistributedDataset(args.data_path, bmt.rank(), bmt.world_size()),
        max_length=max_length - prompt_length,
        prompt_length=prompt_length,
        tokenizer=tokenizer,
    )
    dataloader = BatchPacker(dataset, 512, batch_size=batch_size)
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), weight_decay=0.01)
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr = 0.1, warmup_iter = 2000, end_iter = None, num_iter = 0)
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    optim_manager = bmt.optim.OptimManager(loss_scale=1048576)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    CookTrainer.set_compression(config, model, optimizer, teacher)

    for iteration, data in enumerate(dataloader):

        assert len(data["ctx"]) == batch_size
        input_idx = data["ctx"].int().cuda()
        input_length = data["len_ctx"].int().cuda()
        input_context = data["context"].bool().cuda()
        input_position = data["position"].float().cuda()
        input_segment = data["segment"].int().cuda()
        input_span = data["span"].int().cuda()
        targets = data["tgt"].long().cuda()
        
        # ===========
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # ===========
        outputs = CookTrainer.forward(model, loss_func, targets, input_idx, input_length, input_context, input_position, input_segment, input_span)
        loss = outputs.loss
        global_loss = bmt.sum_loss(loss).item()

        # ===========
        optim_manager.backward(loss)

        if do_distill:
            d_loss = bmt.sum_loss(outputs.d_loss).item()
        else:
            d_loss = 0

        # ===========
        grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=1.0)
        optim_manager.step()

        bmt.print_rank('iter: {} | loss: {:.4f} | grad_norm: {} | d_loss: {}'.format(iteration, global_loss, grad_norm, d_loss))

if __name__ == "__main__":
    main()