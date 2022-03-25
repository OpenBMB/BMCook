
<div align="center">

<h1>🍳 BMCook</h1>

------

<p align="center">
    
<a href='https://bmcook.readthedocs.io/en/main/index.html'>
    <img src='https://readthedocs.org/projects/bmcook/badge/?version=main' alt='Documentation Status' />
</a>

<a href="https://github.com/OpenBMB/BMCook/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/BMCook">
</a>
    
</p>    

</div>


BMCook is a model compression toolkit for large-scale pre-trained language models (PLMs), which integrates multiple model compression methods. You can combine them in any way to achieve the desired speedup.

## Installation

To use BMCook, first install BMTrain.

**From PyPI (Recommend)**

```shell
$ pip install bmtrain
```

**From Source**

```shell
$ git clone https://github.com/OpenBMB/BMTrain.git
$ cd BMTrain
$ python3 setup.py install
```

Please refer to [the installation guide](https://bmtrain.readthedocs.io/en/latest/) of BMTrain for more details.

Then, clone the repository.


```shell
$ git clone git@github.com:OpenBMB/BMCook.git
```

## Examples

The `example` folder provides example codes based on GPT-J (6B).

Quantization-aware training：

```
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt-j-int8 \
     --model gpt-j-full-int8 \
     --start-lr 1e-4 \
     --load gpt-j.bin
```

Quantization-aware training with knowledge distillation：
```
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt-j-int8-distill \
     --model gpt-j-full-int8 \
     --start-lr 1e-4 \
     --load gpt-j.bin \
     --use-kd \
     --kd-mse-last-hidden \
     --kd-loss-scale 1 \
     --load-teacher gpt-j.bin
```

Model pruning：
```
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt-j-prune \
     --model gpt-j-full \
     --start-lr 1e-4 \
     --load gpt-j.bin \
     --use-pruning \
     --use-kd \
     --kd-mse-last-hidden \
     --kd-loss-scale 1 \
     --load-teacher gpt-j.bin
```

Transform the activation function from GeLU to ReLU：
```
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt-j-relu \
     --model gpt-j-full-relu \
     --start-lr 1e-4 \
     --load gpt-j.bin \
     --use-kd \
     --kd-mse-last-hidden \
     --kd-loss-scale 1 \
     --load-teacher gpt-j.bin
```

MoEfication (save the hidden states and then use the MoEfication toolkit)：
```
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt-j-moe \
     --model gpt-j-full-relu \
     --start-lr 1e-4 \
     --load gpt-j-relu.bin \
     --save-hidden
```

Combine quantization, pruning and knowledge distillation：
```
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt-j-int8-prune-distill \
     --model gpt-j-full-int8 \
     --start-lr 1e-4 \
     --load gpt-j.bin \
     --use-pruning \
     --use-kd \
     --kd-mse-last-hidden \
     --kd-loss-scale 1 \
     --load-teacher gpt-j.bin
```

## Implementation

Quantization：
```
    BMQuant.quantize(model)
```

Distillation：
```
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
```

Pruning：
```
    BMPrune.compute_mask(model, m4n2_2d_greedy, checkpoint=args.pruning_mask_path)
    BMPrune.set_optim_for_pruning(optimizer)
```

MoEfication：
```
    BMMoE.moefy(model, args.num_expert, args.topk, checkpoint=args.moe_path)
```

## Comparisons

|                 | Model Quantization | Model Pruning | Knowledge Distillation | Model MoEfication |
|-----------------|--------------------|---------------|------------------------|-------------------|
| [TextPruner](https://github.com/airaria/TextPruner)      |       -             | ✅             |          -              |      -             |
| [TensorFlow Lite](https://www.tensorflow.org/lite) | ✅                  | ✅             |          -              |           -        |
| [PyTorch](https://pytorch.org/)         | ✅                  | ✅             |            -            |          -         |
| [TextBrewer](https://github.com/airaria/TextBrewer)      |           -         | ✅             | ✅                      |         -          |
| BMCook          | ✅                  | ✅             | ✅                      | ✅                 |

