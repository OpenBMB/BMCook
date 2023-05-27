import torch
import bmtrain as bmt
from bmcook.utils.config import ConfigParser
from bmcook import CookTrainer

import sys, os
sys.path.insert(0, os.getcwd())
from cpm_live.arguments import parse_args
from cpm_live.models import CPMAntConfig, CPMAnt
from cpm_live.tokenizers import CPMAntTokenizer
from cpm_live.data import CPMAnt_Dataset, DistributedMMapIndexedDataset, BatchPacker

def main():

    bmt.init_distributed(seed = 233, loss_scale_factor = 2, loss_scale_steps = 512)

    args = parse_args()
    batch_size = 4

    config = CPMAntConfig.from_json_file(args.model_config)
    model = CPMAnt(config)
    bmt.load(model, args.load)

    teacher_config = CPMAntConfig.from_json_file(args.teacher_config)
    teacher = CPMAnt(teacher_config)
    bmt.load(teacher, args.load_teacher)

    config = ConfigParser(args.cook_config)
    

    tokenizer = CPMAntTokenizer('vocabs/ant.txt')
    dataset = CPMAnt_Dataset(
        DistributedMMapIndexedDataset(args.data_path, "cpm_live_text_merge_context", bmt.rank(), bmt.world_size()),
        max_length = 512 - 32, 
        prompt_length = 32,
        tokenizer = tokenizer
    )
    dataloader = BatchPacker(dataset, 512, batch_size=batch_size)
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), 
                                                weight_decay=0.01, 
                                                scale=1048576)
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, 
                                            start_lr = 0.1,
                                            warmup_iter = 2000, 
                                            end_iter = None,
                                            num_iter = 0)
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

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
        task_info = data["task"].long().cuda()
        
        # ===========
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # ===========
        outputs = CPMAntTrainer.forward(model, loss_func, targets, input_idx, input_length, input_context, input_position, input_segment, input_span)
        loss, lag_loss, sparsity, d_loss = outputs.loss, outputs.lag_loss, outputs.sparsity, outputs.d_loss
        global_loss = bmt.sum_loss(loss - d_loss).item()

        # ===========
        loss = optimizer.loss_scale(loss) + lag_loss
        loss.backward()
        
        # ===========
        grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, 4.0, scale = optimizer.scale, norm_type = 2)    
        bmt.optim_step(optimizer, lr_scheduler)

        bmt.print_rank('iter: {} | loss: {:.4f} | grad_norm: {} | d_loss: {} | lag_loss: {:.4f} | sparsity: {:.4f}'.format(iteration, global_loss, grad_norm, d_loss, lag_loss, sparsity))

if __name__ == "__main__":
    main()