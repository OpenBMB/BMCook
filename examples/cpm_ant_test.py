import torch
import bmtrain as bmt
import sys
sys.path.append('/bjzhyai03/gongbt/codebase/BMCook')
from bmcook.utils.config import ConfigParser
from cpm_ant.models import CPMAntConfig, CPMAnt
from cpm_ant.tokenizers import CPMAntTokenizer
from cpm_ant.data import CPMAnt_Dataset, DistributedMMapIndexedDataset, BatchPacker

from bmcook.pruning import BMPrune

bmt.init_distributed(seed = 233, loss_scale_factor = 2, loss_scale_steps = 512)
batch_size = 4

config = CPMAntConfig.from_json_file('/bjzhyai03/cpm_live_compress/ckpt/1B/cpm_live_1B_pruned.json')
model = CPMAnt(config)
bmt.load(model, '/bjzhyai03/cpm_live_compress/ckpt/1B/cpm_live_checkpoint_1B_pruned.pt')

config = ConfigParser('/bjzhyai03/gongbt/codebase/BMCook/examples/cpm_ant/config/bmcook.json')
BMPrune.compute_mask(model, config)

tokenizer = CPMAntTokenizer('/bjzhyai03/cpm_live_compress/vocab/vocab.txt')
dataset = CPMAnt_Dataset(
    DistributedMMapIndexedDataset("/bjzhyai03/cpm_live_compress/data/merged/", "cpm_live_text_merge_context", bmt.rank(), bmt.world_size()),
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

#with torch.no_grad():
for iteration, data in enumerate(dataloader):
    if iteration == 10:
        break

    #if iteration == 5:
    #    model.sprune_plugin['mapping']['ffn'][0]['mask'] = 0.

    #iteration = iteration + 1
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
    logits, _ = model(input_idx, input_length, input_context, input_position, input_segment, input_span)
    loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
    global_loss = bmt.sum_loss(loss).item()

    # ===========
    loss = optimizer.loss_scale(loss)
    #loss.backward()
    
    # ===========
    grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, 4.0, scale = optimizer.scale, norm_type = 2)    
    #bmt.optim_step(optimizer, lr_scheduler)

    bmt.print_rank(iteration, global_loss)