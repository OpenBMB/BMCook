Get Started
=====

.. _introduction:

Introduction
------------

BMCook (https://github.com/OpenBMB/BMCook) is a model compression toolkit for large-scale pre-trained language models (PLMs), which integrates multiple model compression methods. You can combine them in any way to achieve the desired speedup. Specifically, we implement the following four model compression methods:

* **Model Quantization**: Quantization compresses neural networks into smaller size by representing parameters with low bit precision, e.g., 8-bit integer (INT8) instead of 32-bit floating point (FP32). It reduces memory footprint by storing parameters in low precision and accelerate the computation in low precision. In this toolkit, we target quantization-aware training, which simulates the computation in low precision during training to make the model parameter adapt to low precision.

* **Model Pruning**: Pruning compresses neural networks by removing unimportant parameters. According the granularity of pruning, it is categorized into structured pruning and unstructured pruning. In this toolkit, we implement both pruning methods.

* **Knowledge Distillation**: Knowledge distillation aims to alleviate the performance degradation caused by model compression. It provides more informative training objective than the conventional pre-training does. 

* **Model MoEfication**: MoEfication utilize the sparse activation phenomenon in PLMs and split the feed-forward networks into several small expert networks for conditional computation of PLMs.


.. _installation:

Installation
------------

To use BMCook, first install BMTrain using pip:

.. code-block:: console

   $ pip install bmtrain

Then, clone the repository from our github page (don’t forget to star us!)

.. code-block:: console

   $ git clone git@github.com:OpenBMB/BMCook.git


.. _examples:

Examples
------------

In the `example` folder, we provide the example codes based on GPT-J (6B).

Quantization-aware training:

.. code-block:: console

   $ torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt-j-int8 \
     --model gpt-j-full-int8 \
     --start-lr 1e-4 \
     --load gpt-j.bin


Quantization-aware training with knowledge distillation:


.. code-block:: console

   $ torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt-j-int8-distill \
     --model gpt-j-full-int8 \
     --start-lr 1e-4 \
     --load gpt-j.bin \
     --use-kd \
     --kd-mse-last-hidden \
     --kd-loss-scale 1 \
     --load-teacher gpt-j.bin

Model pruning with knowledge distillation:

.. code-block:: console

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

For MoEfication, we first save the hidden states and then split the feed-forward networks:

.. code-block:: console

   $ torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py \
     --save-dir results/gpt-j-moe \
     --model gpt-j-full-relu \
     --start-lr 1e-4 \
     --load gpt-j-relu.bin \
     --save-hidden
   $ python split.py results/gpt-j-moe
   $ python routing.py results/gpt-j-moe

Furthermore, we combine different compression methods (Quantization, Pruning, and Knowledge Distillation):

.. code-block:: console

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