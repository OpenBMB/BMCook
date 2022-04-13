# Quick Start

## Usage of Different Modules

### Quantization

You can use `BMQuant` to enable quantization-aware training as follows:

```
  BMQuant.quantize(model)
```

To use this module, you need to implement the linear transformations with the following format:

```
  # original:
  # torch.nn.functional.linear(input, weight, bias=None)
  import cpm_kernels.torch as ct
  ct.bmm(weight.unsqueeze(0), False, input, False, int8=self.int8)
```

where `weight` is the weight tensor, `input` is the input tensor. Modules have the attribute `int8`, which is a boolean, to control whether the operation is quantized.

### Knowledge Distillation

You can use `BMDistill` to enable knowledge distillation as follows:

```
  BMDistill.set_forward(model, teacher_model, foward_fn)
```

It will modify the forward function to add distillation loss.

Here is an example of the forward function.

```
  def forward(model, dec_input, dec_length, targets, loss_func, 
              output_hidden_states=False):
      outputs = model(
          dec_input, dec_length, output_hidden_states=output_hidden_states)
      logits = outputs[0]
      batch, seq_len, vocab_out_size = logits.size()

      loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))

      return (loss,) + outputs
```

### Weight Pruning

You can use `BMPrune` to enable pruning-aware training as follows:

```
  BMPrune.compute_mask(model, prune_func, checkpoint=pruning_mask_path)
  BMPrune.set_optim_for_pruning(optimizer)
```

`prune_func` is used to compute the pruning mask to construct 2:4 sparsity patterns, which could be accelerated on NVIDIA GPUs. It is recommended to used `m4n2_2d_greedy` which can support the combination with MoEfication. If you only need pruning, you can use `m4n2_1d` instead.

### MoEfication

You can use `BMMoE` to simulate the MoE operation based on the result of MoEfication as follows:

```
  BMMoE.moefy(model, num_expert, topk, checkpoint_path)
```

For more details, please refer to the API documentation.

## Examples Based on GPT-J

In the `example` folder, we provide the example codes based on GPT-J (6B).

### Quantization-aware training

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

### Model pruning

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

### MoEfication

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

### Combination

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