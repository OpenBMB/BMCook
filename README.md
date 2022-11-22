
<div align="center">

<h1><img src="docs/_static/logo.png" height="28px" /> BMCook</h1>

**Model Compression for Big Models**
    
</div>


<p align="center">
  <a href="#overview">Overview</a> • <a href="#documentation">Documentation</a> • <a href="#install">Installation</a> • <a href="#usage">Usage</a> • <a href="#quick-start">Quick Start</a> • <a href="./README-ZH.md" target="_blank">简体中文</a>
<br>
</p>

<p align="center">
	<a href='https://bmcook.readthedocs.io/en/main/'>
	    <img src='https://readthedocs.org/projects/bmcook/badge/?version=main' alt='doc' />
	</a>
	<a href="https://github.com/OpenBMB/BMCook/blob/main/LICENSE">
	    <img alt="github" src="https://img.shields.io/github/license/OpenBMB/BMCook">
	</a>
	<a>
		 <img alt="version" src="https://img.shields.io/badge/version-0.1.0-blue">
	</a>
</p>    

## What's New

- 2022/5/17 Support PLMs in [model-center](https://github.com/OpenBMB/ModelCenter).
- 2022/3/29 (**BMCook 0.1.0**) Now we publicly release the first version of BMCook.

<div id="overview"></div>

## Overview

BMCook is a model compression toolkit for large-scale pre-trained language models (PLMs), which integrates multiple model compression methods. You can combine them in any way to achieve the desired speedup. Specifically, we implement the following four model compression methods, knowledge distillation, model pruning, model quantization, and model MoEfication. It has following features:

- **Various Supported Methods.** Compared to existing compression toolkits, BMCook supports all mainstream acceleration methods for pre-trained language models.
- **User Friendly.** Based on BMCook, we can implement different compression methods with just a few lines of codes.
- **Combination in Any Way.** Due to  the decoupled implications, the compression methods can be combined in any way towards extreme acceleration.

<div id="documentation"></div>

## Documentation
Our [documentation](https://bmcook.readthedocs.io/en/main/) provides more information about the package.

<div id="install"></div>

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

Then, install BMCook. 

**From PyPI (Recommend)**

```shell
$ pip install bmcook
```

**From source**

```shell
$ git clone git@github.com:OpenBMB/BMCook.git
cd BMCook
python3 setup.py install
```

<div id="usage"></div>

## Usage

### 1. Design your BMCook config.
You should give a json file to state your compress strategy.

```json
{ "distillation": {
    "ce_scale": 0,
    "ce_temp": 1,
      
    "mse_hidn_scale": 0,
    "mse_hidn_module": ['[placehold]'],
    "mse_hidn_proj": false,
      
    "mse_att_scale": 0,
    "mse_att_module": ['[placehold]'],
  },

  "pruning": {
    "is_pruning": false,
    "pruning_mask_path": None,
    "pruned_module": ['[placehold]'],
    "mask_method": "m4n2_1d/m4n2_2d/sprune",
    "sprune": {
        "criterion": "l0",
        "training_mask": ['[placehold]'],
        "fixed_mask_path": "",
        "mask_mode": "train_mask",
        "target_sparsity": 0.5
    }
  },

  "quantization": {
    "is_quant": false,
    "quantized_module": [],
  },

  "MoEfication": {
    "is_moefy": false,
    "first_FFN_module": ['[placehold]'],
  }
}
```
To notice:

- `is_moefy`, `is_quant`, `ispruning` are switch parameters. If false, other parameters will be blocked. `mask_method` takes similar works. When `mask_method` is "m4n2_1d" or "m4n2_2d", it will execute unstructure pruning, but when is "sprune", the `sprune` field will be activated. For distillation, when the `ce_scale` or `mse_hidn_scale` is greater than 0, the corresponding distilling mode will be switched on.

- It's not recommended to use MoE and Distilling simultaneously.

### 2. Basic usage in your code.
BMCook provides unified interface `CookTrainer`. BMCook will introduce distillation pruning and MoEfication, which may add some terms to model outputs. You can use it to manage your model, and these modifications.
```python
from bmcook import CookTrainer
from bmcook.utils.config import ConfigParser

#prepare your model, dataloader and optimizer...
...

# setting up your BMCook strategy
CookTrainer.set_compression(cookconfig, model, optimizer, model_distill)

# train
for data in dataloader:
    targets = ...
    ...
    outputs = CookTrainer.forward(model, loss_func, targets, *your_model_inputs, **your_model_kwinputs)

    [loss, model_outputs, lag_loss, sparsity, distill_loss] = outputs
```
the loss equals to the sum of model_loss, lag_loss and distill_loss. So if you wanna know the model performance, please minus them. Noticed that if sprune is not setted, the lag_loss and loss_func will be `None`, so do distilling.
```python
model_loss = loss - lag_loss - distill_loss # sprune and distilling both setted.
model_loss = loss - distill_loss # only distilling used. 
```

BMCook also provides discrete interfaces to initialize compression settings. If you want to design your Trainer for your own needs, you can use these discrete interfaces. Noticed that the output format should keep the same with `CookTrainer` when you define your own Trainer. For details about extension on `CookTrainer`, you can refer to `CPMAntTrainer`.
```python
from bmcook import BMDistill

# Define your own Trainer. 
Trainer = ...

# Set up the distillation
Trainer.forward = BMDistill.set_forward(model, teacher, Trainer.forward, cook_config)
```

### 3. How to run your code
You can run your code as normal, but should state where your cookconfig is:
```shell
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir ... \
     --model ... \
     --start-lr ... \
     --cook-config  ... \ # give your cook config path
```


<div id="quick-start"></div>

## Quick Start

The `examples` folder provides pruning example based on CPM-Live, GPT2-Base, T5-large, please check [examples](https://github.com/OpenBMB/BMCook/blob/main/examples/README.md) for more details.

Take GPT2 as example:

Quantization-aware training：

```
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt2-int8 \
     --model gpt2-base \
     --start-lr 1e-4 \
     --cook-config configs/gpt2-int8.json \
```

Quantization-aware training with knowledge distillation：
```
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt2-int8-kd \
     --model gpt2-base \
     --start-lr 1e-4 \
     --cook-config configs/gpt2-int8-kd.json \
```

Model pruning：
```
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt2-prune \
     --model gpt2-base \
     --start-lr 1e-4 \
     --cook-config configs/gpt2-prune.json \
```
In this case, we only prune the input embedding layer. You can include more modules by changing the `pruned_module` field in the config file.

MoEfication (save the hidden states and then use the MoEfication toolkit)：
```
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt2-moe \
     --model gpt2-base \
     --start-lr 1e-4 \
     --cook-config configs/gpt2-moe.json \
```

Combine quantization, pruning and knowledge distillation：
```
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt2-combine \
     --model gpt2-base \
     --start-lr 1e-4 \
     --cook-config configs/gpt2-combine.json \
```

## Performances

Based on GPT-J, we evaluate different combinations of compression techniques. The corpus is OpenWebText. We also train a small GPT-J with 0.7B parameters based on this corpus from scratch, GPT-J (0.7B).

|                        |     LM Loss    |     Relative Performance    |     Speedup    |
|------------------------|----------------|-----------------------------|----------------|
|     GPT-J              |           3.37 |                        -    |          1x    |
|     GPT-J (0.7B)       |           4.06 |                       83.0% |         ~10x   |
|     GPT-J (P+D)        |           3.57 |                       94.4% |          2x    |
|     GPT-J (P+D+Q)      |           3.58 |                       94.1% |          8x    |
|     GPT-J (P+D+Q+M)    |           3.69 |                       91.3% |          10x   |

D denotes knowledge distillation. P denotes pruning. Q denotes quantization. M denotes MoEfication.

## Comparisons

|                 | Model Quantization | Model Pruning | Knowledge Distillation | Model MoEfication |
|-----------------|--------------------|---------------|------------------------|-------------------|
| [TextPruner](https://github.com/airaria/TextPruner)      |       -             | ✅             |          -              |      -             |
| [TensorFlow Lite](https://www.tensorflow.org/lite) | ✅                  | ✅             |          -              |           -        |
| [PyTorch](https://pytorch.org/)         | ✅                  | ✅             |            -            |          -         |
| [TextBrewer](https://github.com/airaria/TextBrewer)      |           -         | ✅             | ✅                      |         -          |
| BMCook          | ✅                  | ✅             | ✅                      | ✅                 |

## Community
We welcome everyone to contribute codes following our [contributing guidelines](https://github.com/OpenBMB/BMCook/blob/main/CONTRIBUTING.md).

You can also find us on other platforms:
- QQ Group: 735930538
- WeChat Official Account: OpenBMB
- Website: https://www.openbmb.org
- Weibo: http://weibo.cn/OpenBMB
- Twitter: https://twitter.com/OpenBMB

## License

The package is released under the [Apache 2.0](https://github.com/OpenBMB/BMCook/blob/main/LICENSE) License.

## Contributors

We thank Zhengyan Zhang, Baitao Gong, Yingfa Chen, Guoyang Zeng, Jie Zhou, and Zhi Zheng for the contribution. More contributors are welcome!
