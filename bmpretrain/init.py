import torch
import random
import torch.distributed as dist
import os
from .utils import print_dict
from .global_var import config
from . import nccl

def init_distributed(
        seed : int = 0,
        loss_scale_factor : float = 2,
        loss_scale_steps : int = 1024,
        gradient_inspect : bool = False
    ):
    torch.backends.cudnn.enabled = False
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    master = os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"]

    store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]) + 1, world_size, is_master=(rank == 0), wait_for_workers=True)
    torch.cuda.set_device(local_rank)

    config["local_rank"] = local_rank
    config["local_size"] = local_size
    config["rank"] = rank
    config["world_size"] = world_size
    config["calc_stream"] = torch.cuda.current_stream()
    config["load_stream"] = torch.cuda.Stream(priority=-1)
    config['barrier_stream'] = torch.cuda.Stream()
    config["load_event"] = torch.cuda.Event()

    config["loss_scale_factor"] = loss_scale_factor if loss_scale_factor > 1 else 1 / loss_scale_factor
    config["loss_scale_steps"] = loss_scale_steps
    config["gradient_inspect"] = gradient_inspect

    cpus_this_worker = None
    
    all_available_cpus = sorted(list(os.sched_getaffinity(0)))

    cpus_per_worker = len(all_available_cpus) // local_size
        
    if cpus_per_worker < 1:
        cpus_this_worker = all_available_cpus
        torch.set_num_threads(1)
    else:
        cpus_this_worker = all_available_cpus[local_rank * cpus_per_worker : (local_rank + 1) * cpus_per_worker]
        os.sched_setaffinity(0, cpus_this_worker)
        torch.set_num_threads( len(cpus_this_worker) )

    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    
    if rank == 0:
        unique_id : bytes = nccl.getUniqueId()
        store.set("BMPRETRAIN_UNIQUE_ID", unique_id.hex() )
    
    unique_id = bytes.fromhex(store.get("BMPRETRAIN_UNIQUE_ID").decode())
    config['comm'] = nccl.commInitRank(unique_id, world_size, rank)
    
    print_dict("Initialization", {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "local_size": local_size,
        "master" : master,
        "device": torch.cuda.current_device(),
        "cpus": cpus_this_worker 
    })
