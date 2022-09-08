====================
Pruning
====================

BMPrune
==========================================

Here is the example configuration for BMPrune:

"pruning": {
   "is_pruning": true, "pruning_mask_path": "prune_mask.bin",
   "pruned_module": ["ffn.ffn.w_in.w.weight", "ffn.ffn.w_out.weight", "input_embedding"],
   "mask_method": "m4n2_1d"
}

Practitioners can turn on pruning by `is_pruning`. The pruning mask is stored in `pruning_mask_path`. The pruned modules are specified by `pruned_module`. To simplify the list, practitioners can only provide the suffix of the modules. The mask method `mask_method`` is to choose the algorithm for the computation of the pruning mask.

.. autoclass:: pruning.BMPrune
   :members:
