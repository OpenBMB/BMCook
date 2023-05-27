<div align="center">

<h1>🍳 BMCook</h1>

**大模型压缩工具包**

</div>

<p align="center">
  <a href="#overview">总览</a> • <a href="#documentation">文档</a> • <a href="#install">安装</a> • <a href="#quick-start">快速上手</a> • <a href="./README.md" target="_blank">English</a>
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

## 最新动态

- 2023/5/27 支持Decoder-only模型的结构化剪枝。其中包括对[CPM-Live](https://github.com/OpenBMB/CPM-Live/tree/master)系列模型的压缩。
- 2022/5/17 支持[model-center](https://github.com/OpenBMB/ModelCenter)中的预训练模型压缩。
- 2022/3/20 (BMCook 0.1.0) 第一版BMCook发布了！

<div id="overview"></div>

## 总览

BMCook是一个用于大规模预训练语言模型（PLM）的模型压缩工具包，它集成了多种模型压缩方法。你可以以任何方式组合它们，以满足特定的计算需求。具体来说，本工具包实现了以下四种模型压缩方法：知识蒸馏、模型剪枝、模型量化和模型专家化。

- **支持多种方法** 与现有的压缩工具包相比，BMCook支持所有主流的预训练语言模型加速方法。
- **用户友好** 基于BMCook，用户只需几行代码就可以实现不同的压缩方法。
- **任意组合** 受益于解耦合的实现方式，不同方法可以任意组合以追求极致压缩。

<div id="documentation"></div>

## 文档
我们的[文档](https://bmcook.readthedocs.io/en/main/)提供了关于该工具包的更多信息。

<div id="install"></div>

## 安装

BMCook基于BMTrain进行开发，使用前需先安装BMTrain

**从PyPI安装（推荐）**

```shell
$ pip install bmtrain
```

**从源代码安装**

```shell
$ git clone https://github.com/OpenBMB/BMTrain.git
$ cd BMTrain
$ python3 setup.py install
```

更多细节请请参考[BMTrain](https://bmtrain.readthedocs.io/en/latest/)的安装指南。

安装完BMTrain后，再拉取本仓库。

```shell
$ git clone git@github.com:OpenBMB/BMCook.git
```

<div id="quick-start"></div>

## 快速上手

`cpm_live_example` 文件夹提供了 CPM-Live 结构化剪枝的样例，详见[介绍](https://github.com/OpenBMB/BMCook/blob/main/cpm_live_example/README.md).


`gpt-example`文件夹提供了基于 Model Center 中 GPT2-Base 的样例代码。

模型量化：

```
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt2-int8 \
     --model gpt2-base \
     --start-lr 1e-4 \
     --cook-config configs/gpt2-int8.json \
```

在训练过程中加入模型蒸馏：
```
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt2-int8-kd \
     --model gpt2-base \
     --start-lr 1e-4 \
     --cook-config configs/gpt2-int8-kd.json \
```

模型剪枝：
```
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt2-prune \
     --model gpt2-base \
     --start-lr 1e-4 \
     --cook-config configs/gpt2-prune.json \
```
该配置文件只对输入层进行了剪枝，你可以通过修改配置文件中的`prune_module`来引入更多模块。

模型专家化（不需要训练，只需保存中间计算结果）：
```
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt2-moe \
     --model gpt2-base \
     --start-lr 1e-4 \
     --cook-config configs/gpt2-moe.json \
```

与此同时，不同的压缩方法可以任意组合，以下是量化、剪枝和蒸馏结合的样例代码：
```
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt2-combine \
     --model gpt2-base \
     --start-lr 1e-4 \
     --cook-config configs/gpt2-combine.json \
```

## 压缩效果

基于T5-3B，我们评估了不同的压缩组合，压缩语料库使用了Pile。选择SST-2、MNLI和SQuAD作为下游任务进行评测。适配下游任务时，我们固定了预训练模型参数，采用adapter-tuning进行训练。

|                        |     Average Performance    |     Relative Performance    |     Speedup    |
|------------------------|----------------|-----------------------------|----------------|
|     T5-3B              |           0.9258 |                        -    |          1x    |
|     T5-Base       |           0.8796 |                       95.0% |         7x   |
|     T5-3B (P+D)        |           0.9150 |                       98.8% |          2x    |
|     T5-3B (P+D+Q)      |           0.9126 |                       98.6% |          8x    |
|     T5-3B (P+D+Q+M)    |           0.9017 |                       97.4% |          12x   |

D 代表知识蒸馏；P 代表模型剪枝；Q 代表模型量化；M 代表模型专家化。


## 功能对比

|                 | Model Quantization | Model Pruning | Knowledge Distillation | Model MoEfication |
|-----------------|--------------------|---------------|------------------------|-------------------|
| [TextPruner](https://github.com/airaria/TextPruner)      |       -             | ✅             |          -              |      -             |
| [TensorFlow Lite](https://www.tensorflow.org/lite) | ✅                  | ✅             |          -              |           -        |
| [PyTorch](https://pytorch.org/)         | ✅                  | ✅             |            -            |          -         |
| [TextBrewer](https://github.com/airaria/TextBrewer)      |           -         | ✅             | ✅                      |         -          |
| BMCook          | ✅                  | ✅             | ✅                      | ✅                 |

## 开源社区

欢迎贡献者参照我们的[贡献指南](https://github.com/OpenBMB/BMCook/blob/main/CONTRIBUTING.md)贡献相关代码。

您也可以在其他平台与我们沟通交流:
- QQ群: 735930538
- 微信公众号: OpenBMB
- 官方网站: https://www.openbmb.org
- 微博: http://weibo.cn/OpenBMB
- Twitter: https://twitter.com/OpenBMB

## 开源许可

该工具包使用[Apache 2.0](https://github.com/OpenBMB/BMCook/blob/main/LICENSE)开源许可证。

