from typing import Optional
import torch
from .global_var import config
from .utils import print_rank
from .lr_scheduler.warmup import WarmupLRSchduler


def optim_step(optim : torch.optim.Optimizer, lr_scheduler : Optional[WarmupLRSchduler] = None):
    """
    Backward with loss scale.
    Synchronize streams before optimizer steps.
    """
    
    has_scale = hasattr(optim, 'scale')
    current_stream =  torch.cuda.current_stream()
    # some reduce ops of distributed parameter were launched on load stream
    current_stream.wait_stream(config['load_stream'])

    if has_scale:
        try:
            optim.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
        except OverflowError:
            print_rank("Gradient overflow, change scale from %lf to %lf" % (optim.scale, optim.scale / config["loss_scale_factor"]))
            optim.justify_scale(optim.scale / config["loss_scale_factor"])
            optim.zero_grad()

        if optim.steps_since_last_scale >= config["loss_scale_steps"]:
            optim.justify_scale(optim.scale * config["loss_scale_factor"])
    else:
        optim.step()

    config['load_stream'].wait_stream(current_stream)