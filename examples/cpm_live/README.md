# BMCook SPrune Guide for CPM-Live Models

Suppose you have obtained the model code from [CPM-Bee](https://github.com/OpenBMB/CPM-Bee) or [CPM-Live](https://github.com/OpenBMB/CPM-Live), and has known how to train and fine-tune the CPM-Live models.

Here is a brief training script with some pseudocode of CPM-Bee:
```python
from cpm_live.models import CPMBee, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer

# prepare the model
config = CPMBeeConfig.from_json_file(args.model_config)
model = CPMBee(config)

# prepare optimizer
optimizer = init_an_optimizer
lr_scheduler = init_a_lr_schedular
optim_manager = init_an_optim_manager
optim_manager.add_optimizer(optimizer, lr_scheduler)

# training loop
for data, target in your_dataloader:
    optim_manager.zero_grad()

    logits, _ = model(*data)

    loss = your_loss_func(logits, target)

    optim_manager.backward(loss)

    optim_manager.step()
```

You just need to add a few lines of code:

```python
import bmcook
from bmcook import CookTrainer, ConfigParser
from cpm_live.models import CPMBee, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer

# prepare the model
config = CPMBeeConfig.from_json_file(args.model_config)
model = CPMBee(config)

# prepare optimizer
optimizer = init_an_optimizer
lr_scheduler = init_a_lr_schedular
optim_manager = init_an_optim_manager
optim_manager.add_optimizer(optimizer, lr_scheduler)

# compression setup
teacher = your_teacher_model
cook_config = ConfigParser(your_cook_config_path)
CookTrainer.set_compression(cook_config, model, optimizer, teacher=teacher, quant_layer_cls=Linear)

# training loop
for data, target in your_dataloader:
    optim_manager.zero_grad()

    # use bmcook
    outputs = CookTrainer.forward(model, your_loss_func, target, **data)
    loss = output.loss

    optim_manager.backward(loss)

    optim_manager.step()


# bmcook.save_masks(args.cook_mask_save)  # use this to save sprune mask in training
bmcook.save(model, save_name, mode="quant")  # use this to save compressed model, choose mode == "quant" or "prune" to save quantized model or pruned model.
```
Besides, you should prepare a configuration file for compression. Here is an example of structured pruning:
```json
{
    "distillation": {
        "ce_scale": 0,
        "ce_temp": 5,

        "mse_hidn_scale": 0,
        "mse_hidn_module": ["[post]encoder.output_layernorm:[post]encoder.output_layernorm"],
        "mse_hidn_proj": false
    },

    "pruning": {
        "is_pruning": true,
        "pruning_mask_path": "prune_example.bin",
        "pruned_module": ["input_embedding"],
        "mask_method": "sprune",
        "sprune": {
                "criterion": "l0",
                "training_mask": ["att", "ffn"],
                "mask_path": "results/masks/mask.bin",
                "target_mode": "sparsity",
                "is_training": true,
                "target_sparsity": 0.5,
                "start_sparsity": 0.1,
                "hard_binarize": true,
                "tuning": {
                    "iterative": false,
                    "interval": 100,
                    "ratio": 0.1
                }
                }
    },

    "quantization": {
        "is_quant": false
    },

    "MoEfication": {
        "is_moefy": false,
        "first_FFN_module": ["ffn.layernorm_before_ffn"]
    }
}
```

To notice that, 
- choose `pruning:is_pruning` as `true` to open pruning; 
- choose `pruning:mask_method` as `sprune` to open structured pruning;
- choose `pruning:sprune:is_training` as `true` to train the mask; 
- choose `pruning:sprune:is_training` as `false` to fix the mask and tune the model