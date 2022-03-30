# Examples

In the `example` folder, we provide the example codes based on GPT-J (6B).

## Quantization-aware training

Quantization-aware training:

```
$ torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
  --save-dir results/gpt-j-int8 \
  --model gpt-j-full-int8 \
  --start-lr 1e-4 \
  --load gpt-j.bin
```


Quantization-aware training with knowledge distillation:


```
$ torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
  --save-dir results/gpt-j-int8-distill \
  --model gpt-j-full-int8 \
  --start-lr 1e-4 \
  --load gpt-j.bin \
  --use-kd \
  --kd-mse-last-hidden \
  --kd-loss-scale 1 \
  --load-teacher gpt-j.bin
```

## Model pruning

Model pruning with knowledge distillation:

```
$ torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
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

## MoEfication

Transform the activation function from GeLU to ReLU：

```
$ torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
  --save-dir results/gpt-j-relu \
  --model gpt-j-full-relu \
  --start-lr 1e-4 \
  --load gpt-j.bin \
  --use-kd \
  --kd-mse-last-hidden \
  --kd-loss-scale 1 \
  --load-teacher gpt-j.bin
```

For MoEfication, we first save the hidden states and then split the feed-forward networks:

```
$ torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
  --save-dir results/gpt-j-moe \
  --model gpt-j-full-relu \
  --start-lr 1e-4 \
  --load gpt-j-relu.bin \
  --save-hidden
$ python moefication/param_cluster_example.py \
  --model_path gpt-j-relu.bin \
  --res_path results/gpt-j-moe \
  --num-layer 28 \
  --num-expert 512 \
  --templates dec_layers.{}.ff.fc_in_weight
$ python moefication/mlp_select_example.py \
  --model_path gpt-j-relu.bin \
  --res_path results/gpt-j-moe \
  --num-layer 28 \
  --num-expert 512 \
  --templates dec_layers.{}.ff.fc_in_weight
```

Please refer to the repo of [MoEfication](https://github.com/thunlp/MoEfication) for more details.

## Combination

Furthermore, we combine different compression methods (Quantization, Pruning, and Knowledge Distillation):

```
$ torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
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