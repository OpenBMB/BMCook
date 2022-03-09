import torch
import sys
import math

mapper = {
    "transformer.ln_f.weight": "layernorm_after_dec.weight",
    "transformer.ln_f.bias": "layernorm_after_dec.bias",
    "lm_head.weight": "lm_head.weight",
    "lm_head.bias": "lm_head.bias",
    "transformer.wte.weight": "input_embedding.weight",
}

ckpt = torch.load(sys.argv[1])
d = {}
for k in ckpt:
    if 'transformer.h.' in k:
        # layers
        if "ln_1" in k:
            # weight bias
            new_k = k.replace("transformer.h.", "dec_layers.")
            d[new_k] = ckpt[k]
        elif "mlp.fc_out.weight" in k:
            new_k = k.replace("transformer.h.", "dec_layers.").replace("mlp.fc_out.weight", "ff.fc_out_weight")
            d[new_k] = ckpt[k].reshape(16384, 4096).transpose(0, 1) #* math.sqrt(16384)
        elif "mlp.fc_out.bias" in k:
            ckpt[k] = ckpt[k].unsqueeze(0).unsqueeze(-1)
            new_k = k.replace("transformer.h.", "dec_layers.").replace("mlp.fc_out.bias", "ff.fc_out_bias")
            d[new_k] = ckpt[k]
        elif "mlp.fc_in.weight" in k:
            new_k = k.replace("transformer.h.", "dec_layers.").replace("mlp.fc_in.weight", "ff.fc_in_weight")
            d[new_k] = ckpt[k].reshape(4096, 16384).transpose(0, 1) #* math.sqrt(4096)
        elif "mlp.fc_in.bias" in k:
            ckpt[k] = ckpt[k].unsqueeze(0).unsqueeze(-1)
            new_k = k.replace("transformer.h.", "dec_layers.").replace("mlp.fc_in.bias", "ff.fc_in_bias")
            d[new_k] = ckpt[k]
        elif "attn.out_proj.weight" in k or "attn.k_proj.weight" in k or "attn.v_proj.weight" in k or "attn.q_proj.weight" in k:
            new_k = k.replace("transformer.h.", "dec_layers.").replace("attn", "self_attention")[:-7]
            d[new_k] = ckpt[k].contiguous() #* math.sqrt(4096)
        elif "attn.masked_bias" in k or "attn.bias" in k:
            pass
        else:
            assert False
    elif k in mapper:
        if k == "lm_head.bias":
            ckpt[k] = ckpt[k].unsqueeze(0).unsqueeze(-1)
        elif k == "lm_head.weight":
            ckpt[k] = ckpt[k].contiguous()
        d[mapper[k]] = ckpt[k]
    else:
        assert False

torch.save(d, sys.argv[2])
