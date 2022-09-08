====================
Distillation
====================

BMDistill
==========================================

Here is the example configuration for BMDistill:

"distillation": {
        "ce_scale": 0,
        "mse_hidn_scale": 1,
        "mse_hidn_module": ["[post]encoder.output_layernorm:[post]encoder.output_layernorm", "[post]decoder.output_layernorm:[post]decoder.output_layernorm"],
        "mse_hidn_proj": false
}

Currently, BMCook supports two kinds of distillation objectives, KL divergence between output distributions (turn on when `ce_scale>0`) and mean squared error (MSE) between hidden states (turn on when `mse_hidn_scale>0`). Practitioners need to specify the hidden states used for MSE by `mse_hidn_module`. Meanwhile, the dimensions of the hidden states may be different between teacher and student models. Therefore, the hidden states of the teacher model need to be projected to the same dimension as those of the student model.Practitioners can turn on `mse_hidn_proj` for simple linear projection.

.. autoclass:: distilling.BMDistill
   :members:

.. autoclass:: distilling.get_module_info

.. autofunction:: distilling.get_module_map

.. autofunction:: distilling.update_forward