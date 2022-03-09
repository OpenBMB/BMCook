import torch
from typing_extensions import TypedDict
class ConfigMap(TypedDict):
    rank : int
    local_rank : int
    world_size : int
    local_size : int

    calc_stream : torch.cuda.Stream
    load_stream : torch.cuda.Stream
    load_event : torch.cuda.Event
    barrier_stream : torch.cuda.Stream

    loss_scale_factor : float
    loss_scale_steps : int

    gradient_inspect : bool

    comm : 'NCCLCommunicator'

config = ConfigMap()

def rank():
    return config['rank']

def world_size():
    return config['world_size']