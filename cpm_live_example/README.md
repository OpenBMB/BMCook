# BMCook SPrune Guide for CPM-Ant Compression
## Overview
BMCook Sprune is a main tool to compress CPM-Ant. The compression process is task-agnostic, in pre-training stage and shares the pre-training dataset. If you want to compress CPM-Ant for a specific downstream task, just prepare your task dataset in the way as [CPM-Ant](https://github.com/OpenBMB/CPM-Live/tree/master/cpm-live#4-feed-your-data) implements and follow the same compression process.

BMCook Sprune now supports two pruning granularities: layer level and head/dim level. Generally, you can set the number of layers you want for layer level pruning, then set the num_heads or dim_ff to do head/dim level pruning.

More general pruning toolkit will be released soon!


## Quick Start
### Run Puning Script:
```shell
$ bash scripts/prune_cpm_ant.sh
```

### Set Overall BMCook Config
You can switch to prune or not by setting "pruning" in config/bmcook.json as follows:
```json
"pruning": {
        "is_pruning": true,
        "pruning_mask_path": "your unstructure mask path",
        "pruned_module": ["ffn.ffn.w_in.w.weight", 
                            "ffn.ffn.w_out.weight", 
                            "self_attention.project_q.weight",
                            "self_attention.project_k.weight", 
                            "self_attention.project_v.weight", 
                            "self_attention.attention_out.weight"],
        "mask_method": "fine-grained"
    },
```
If you want prune the model, please set "is_pruning" == True.

The "mask_method" will decide:

- coarse-grained: layer-level structure pruning.
- fine-grained: head/dim-level structure pruning.

### Set Specific SPrune Config
You can set up your own pruning scheme by modifying config/l0_pruning.json as follows:
```json
{
    "coarse-grained": {
        "train_mask": false,
        "coarse_mask": "results/sprune/2B_layer_mask.pt",
        "target_sparsity": 0.67
    },

    "fine-grained": {
        "train_mask": true,
        "coarse_mask": "results/sprune/2B_layer_mask.pt",
        "num_heads": 16,
        "dim_ff": 700,
        "target_num_heads": 4,
        "target_dimff": 350,
        "heads_mask": "results/sprune/300M_head_mask.pt",
        "dimff_mask": "results/sprune/300M_dimff_mask.pt",
        "prune_mode": "att"
    }
}
```
The "train_mask" decides:
- True: The mask will be updated by learning.
- False: The mask will be fixed. it always be used in finetune scene after binarizing the mask.

The "prune_mode" decides:
- att: train the head mask.
- ffn: train the dim_ff mask. 

If you train head mask first, you should provides the fixed mask in "heads_mask" when training fixed mask and vice versa. 

If you still not do actual prune(reshape the model) in layer_level, "coarse_mask" should be provided. It will tell the program which layers has been pruned and will be saved in the layer pruning process. If you already do actual prune, you can omit the "coarse_mask"

### Do Actual Prune
just run the python file:
```shell
$ python do_prune.py
```

It will actully prune the model according to masks provided.


## Example: 1B model -> 300M model
Scheme: num_heads: 16 -> 4, dim_ff: 700 -> 350
1. train att mask
```json
    "fine-grained": {
            "train_mask": true,
            ...
            "target_num_heads": 4,
            "target_dimff": 350,
            "prune_mode": "att"
        }
```
2. fine-tune att mask:
```json
    "fine-grained": {
            "train_mask": false,
            "heads_mask": "results/sprune/300M_head_mask.pt",
            ...
        }
```
3. train ffn mask:
```json
    "fine-grained": {
            "train_mask": true,
            ...
            "prune_mode": "ffn"
        }
```
4. fine-tune ffn mask:
```json
    "fine-grained": {
            "train_mask": false,
            "heads_mask": "results/sprune/300M_head_mask.pt",
            "dimff_mask": "results/sprune/300M_dimff_mask.pt",
            ...
        }
```
5. do actual prune:
```shell
python do_prune.py
```
