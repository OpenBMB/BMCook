# 使用方法

BMCook是一个基于[BMTrain](https://github.com/OpenBMB/BMTrain)开发的模型加速工具包，支持多种模型加速方法，包括模型量化、模型蒸馏、模型剪枝和模型专家化。

## 使用样例

`example`文件夹提供了基于GPT-J（6B参数）的样例代码。

模型量化：

```
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt-j-int8 \
     --model gpt-j-full-int8 \
     --start-lr 1e-4 \
     --load gpt-j.bin
```

在训练过程中加入模型蒸馏：
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

模型剪枝：
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

为了模型专家化，需要把模型激活函数进行一个转换适配：
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

模型专家化（不需要训练，只需保存中间计算结果）：
```
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt-j-moe \
     --model gpt-j-full-relu \
     --start-lr 1e-4 \
     --load gpt-j-relu.bin \
     --save-hidden
```

与此同时，不同的压缩方法可以任意组合，以下是量化、剪枝和蒸馏结合的样例代码：
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

## 代码实现

模型量化，调整bmm中的int8开关即可开启模型量化训练：
```
    ct.bmm(w_0.unsqueeze(0), False, x, False, int8=int8)
```

模型蒸馏，通过修改forward函数加入蒸馏的loss：
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

模型剪枝，首先计算可以丢弃的参数，然后修改优化器：
```
    BMPrune.compute_mask(model, m4n2_2d_greedy, checkpoint=args.pruning_mask_path)
    BMPrune.set_optim_for_pruning(optimizer)
```

模型专家化，修改模型中FFN的计算过程：
```
    BMMoE.moefy(model, args.num_expert, args.topk, checkpoint=args.moe_path)
```

## 功能对比

|                 | Model Quantization | Model Pruning | Knowledge Distillation | Model MoEfication |
|-----------------|--------------------|---------------|------------------------|-------------------|
| [TextPruner](https://github.com/airaria/TextPruner)      |                    | ✅             |                        |                   |
| [TensorFlow Lite](https://www.tensorflow.org/lite) | ✅                  | ✅             |                        |                   |
| [PyTorch](https://pytorch.org/)         | ✅                  | ✅             |                        |                   |
| [TextBrewer](https://github.com/airaria/TextBrewer)      |                    | ✅             | ✅                      |                   |
| BMCook          | ✅                  | ✅             | ✅                      | ✅                 |

