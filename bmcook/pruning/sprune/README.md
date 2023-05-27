# BMCook SPrune

## What's New
- 2023/5/27 
    1. support the compression of CPM-Bee.
    2. provide brief configuration for **training with sprun mask** and **finetuning with sprune mask**.

## Overview
Structure pruning toolkit of BMCook.

If you want to prune a large PLM in a structured way, BMCook SPrune may help to get a desirable subtructure, based on L0 regularization and lagrangian method. For details, refer to:

1. [Learning Sparse Neural Networks through L_0 Regularization](https://openreview.net/forum?id=H1Y8hhg0b)

2. [Structured Pruning of Large Language Models](https://arxiv.org/abs/1910.04732)

BMCook SPrune use two classes to manage structure pruning process: **SPrunePlugin** and **SPruneEngine**.

**SPrunePlugin** consists of the mask, and directly related to the model. It's responsible for pruning. **SPruneEngine** manage the mask with specific strategy and criterion. It's responsible for finding an appropriate mask.

## Usage

1. Initialize a plugin from the target compressing model:
```python
plugin = SPrunePlugin(model)
```
2. Initialize an engine from the plugin and config:
```python
engine = SPruneEngine(sprune_config, plugin)
```

The process is coupled with BMCook.pruning. You can find it in "pruning/\_\_init\_\_.py"

## Future

We will provide more sprune strategies, like iterational pruning and hidden pruning, in the future. Stay tuned！ 