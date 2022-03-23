Introduction
=====

BMCook (https://github.com/OpenBMB/BMCook) is a model compression toolkit for large-scale pre-trained language models (PLMs), which integrates multiple model compression methods. You can combine them in any way to achieve the desired speedup. Specifically, we implement the following four model compression methods:

* **Model Quantization**: Quantization compresses neural networks into smaller sizes by representing parameters with low-bit precision, e.g., 8-bit integer (INT8) instead of 32-bit floating point (FP32). It reduces memory footprint by storing parameters in low precision and accelerating the computation in low precision. In this toolkit, we target quantization-aware training, which simulates the computation in low precision during training to make the model parameter adapt to low precision.

* **Model Pruning**: Pruning compresses neural networks by removing unimportant parameters. According to the granularity of pruning, it is categorized into structured pruning and unstructured pruning. In this toolkit, we implement both pruning methods.

* **Knowledge Distillation**: Knowledge distillation aims to alleviate the performance degradation caused by model compression. It provides a more informative training objective than the conventional pre-training does. 

* **Model MoEfication**: MoEfication utilizes the sparse activation phenomenon in PLMs and splits the feed-forward networks into several small expert networks for conditional computation of PLMs.