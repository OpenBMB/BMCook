<div align="center">

<h1>♾ BMInf</h1>

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

- 2022/3/20 BMCook正式发布了！

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

## 开源社区

欢迎贡献者参照我们的[贡献指南](https://github.com/OpenBMB/BMCook/blob/main/CONTRIBUTING.md)贡献相关代码。

您也可以在其他平台与我们沟通交流:
- QQ群: 735930538
- 官方网站: http://www.openbmb.org
- 微博: http://weibo.cn/OpenBMB
- Twitter: https://twitter.com/OpenBMB

## 开源许可

该工具包使用[Apache 2.0](https://github.com/OpenBMB/BMCook/blob/main/LICENSE)开源许可证。

## 功能对比

|                 | Model Quantization | Model Pruning | Knowledge Distillation | Model MoEfication |
|-----------------|--------------------|---------------|------------------------|-------------------|
| [TextPruner](https://github.com/airaria/TextPruner)      |       -             | ✅             |          -              |      -             |
| [TensorFlow Lite](https://www.tensorflow.org/lite) | ✅                  | ✅             |          -              |           -        |
| [PyTorch](https://pytorch.org/)         | ✅                  | ✅             |            -            |          -         |
| [TextBrewer](https://github.com/airaria/TextBrewer)      |           -         | ✅             | ✅                      |         -          |
| BMCook          | ✅                  | ✅             | ✅                      | ✅                 |

