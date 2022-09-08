# Quick Start

## Usage of Different Modules

### Configuration

The compression strategy is defined in the configuration file. Here is an example of the configuration file:

```
{
    "distillation": {
        "ce_scale": 0,
        "mse_hidn_scale": 1,
        "mse_hidn_module": ["[post]encoder.output_layernorm:[post]encoder.output_layernorm", "[post]decoder.output_layernorm:[post]decoder.output_layernorm"],
        "mse_hidn_proj": false
    },
    "pruning": {
        "is_pruning": true, "pruning_mask_path": "prune_mask.bin",
        "pruned_module": ["ffn.ffn.w_in.w.weight", "ffn.ffn.w_out.weight", "input_embedding"],
        "mask_method": "m4n2_1d"
    },
    "quantization": { "is_quant": true},
    "MoEfication": {
        "is_moefy": false,
        "first_FFN_module": ["ffn.layernorm_before_ffn"]
    }
}
```

Please refer to the API documentation for the detailed explanation of each parameter.

### Quantization

You can use `BMQuant` to enable quantization-aware training as follows:

```
  BMQuant.quantize(model, config)
```

### Knowledge Distillation

You can use `BMDistill` to enable knowledge distillation as follows:

```
  BMDistill.set_forward(model, teacher_model, foward_fn, config)
```

It will modify the forward function to add distillation loss.

Here is an example of the forward function.

```
  def forward(model, enc_input, enc_length, dec_input, dec_length, targets, loss_func, 
              output_hidden_states=False):
      outputs = model(
          enc_input, enc_length, dec_input, dec_length, output_hidden_states=output_hidden_states)
      logits = outputs[0]
      batch, seq_len, vocab_out_size = logits.size()

      loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))

      return (loss,) + outputs
```

### Weight Pruning

You can use `BMPrune` to enable pruning-aware training as follows:

```
  BMPrune.compute_mask(model, config)
  BMPrune.set_optim_for_pruning(optimizer)
```

### MoEfication

You can use `BMMoE` to get the hidden states for MoEfication:

```
  BMMoE.get_hidden(model, config, Trainer.forward)
```

For more details, please refer to the API documentation.

## Examples Based on CPM-Live

In the `cpm_live_example` folder, we provide the example codes based on CPM-Live.
