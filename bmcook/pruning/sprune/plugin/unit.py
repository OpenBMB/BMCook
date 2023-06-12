import torch
import types
import bmtrain as bmt

from ..utils import get_dim_ff, get_dim_model


class SPruneUnit(object):
    def __init__(
        self,
        name: str, 
        list_in: str,
        dim: int = 1,
        ):
        self.name = name
        self.list_in = list_in
        self.dim = dim
        self.mask = torch.ones(self.dim, dtype=torch.half) 
        self.density = 1 * dim
        self.training: bool = False
    
    def set_training(self):
        self.mask = torch.ones(self.dim, dtype=torch.half)  # 1-dimentional tensor
        self.density = torch.ones(self.dim, dtype=torch.float).sum()  # scalar
        self.training = True


class HeadParamUnit(SPruneUnit):
    def __init__(self, num_heads, dim_head, dim_model, name: str):
        super().__init__(name, list_in="att_params")
        self.dim_head = SPruneUnit(name, "dim_head", dim_head)
        self.num_heads = SPruneUnit(name, "num_heads", num_heads)
        self.dim_model = SPruneUnit(name, "dim_model", dim_model)

    def get_param_exp(self):
        return self.num_heads.density * self.dim_head.density * self.dim_model.density

    def get_param_all(self):
        return self.num_heads.dim * self.dim_head.dim * self.dim_model.dim
    
    def is_training(self):
        return self.dim_head.training | self.num_heads.training | self.dim_model.training

    def set_pruning(self, root: torch.nn.Module):
        if self.is_training():
            for module in [root.project_q, root.project_k, root.project_v]:
                module.forward_unprune = module.forward
                def prune_forward_in(module_self, *args, **kwargs):
                    num_heads, dim_head = self.num_heads.dim, self.dim_head.dim
                    out = module_self.forward_unprune(*args, **kwargs)  # (batch, len, num_heads*dim_head)
                    old_size = out.size()
                    out = out.view(old_size[0], old_size[1], num_heads, dim_head) \
                                * self.num_heads.mask[None, None, :, None].to(out.device) \
                                * self.dim_head.mask[None, None, None, :].to(out.device)
                    out = out.view(*(old_size))
                    return out
                module.forward = types.MethodType(prune_forward_in, module)

            for module in [root.attention_out]:
                module.forward_unprune = module.forward
                def prune_forward_out(module_self, x, *args, **kwargs):
                    if self.dim_model.mask.device != x.device:
                        self.dim_model.mask = self.dim_model.mask.to(x.device)
                    num_heads, dim_head = self.num_heads.dim, self.dim_head.dim
                    old_size = x.size()  # (batch, len, num_heads * dim_head)
                    x = x.view(old_size[0], old_size[1], num_heads, dim_head)
                    x = x \
                        * self.num_heads.mask[None, None, :, None].to(x.device) \
                        * self.dim_head.mask[None, None, None, :].to(x.device)
                    x = x.view(old_size[0], old_size[1], num_heads * dim_head)

                    out = module_self.forward_unprune(x, *args, **kwargs)  # (batch, len, dim_model)
                    out = out * self.dim_model.mask
                    return out
                module.forward = types.MethodType(prune_forward_out, module)

    def prune_params(self, name: str, tensor: torch.Tensor):
        old_num_heads = self.num_heads.dim
        old_dim_head = self.dim_head.dim
        old_dim_model = self.dim_model.dim
        num_heads_indices = torch.nonzero(self.num_heads.mask == 1., as_tuple=True)[0]
        dim_head_indices = torch.nonzero(self.dim_head.mask == 1., as_tuple=True)[0]
        dim_model_indices = torch.nonzero(self.dim_model.mask == 1., as_tuple=True)[0]
        new_num_heads_dim = num_heads_indices.size(0)
        new_dim_head_dim = dim_head_indices.size(0)
        new_dim_model_dim = dim_model_indices.size(0)
        if ".attention_out.weight" not in name:
            old_tensor = tensor.view(old_num_heads, old_dim_head, old_dim_model)
            new_tensor = old_tensor[num_heads_indices, :, :][:, dim_head_indices, :][:, :, dim_model_indices]
            new_tensor = new_tensor.view(new_num_heads_dim * new_dim_head_dim, new_dim_model_dim)
        else:
            old_tensor = tensor.permute(1, 0).contiguous().view(old_num_heads, old_dim_head, old_dim_model)
            new_tensor = old_tensor[num_heads_indices, :, :][:, dim_head_indices, :][:, :, dim_model_indices]
            new_tensor = new_tensor.view(new_num_heads_dim * new_dim_head_dim, new_dim_model_dim).permute(1, 0).contiguous()
        return new_tensor


class FFParamUnit(SPruneUnit):
    def __init__(self, dim_ff: int, dim_model: int, name: str):
        super().__init__(name, list_in="ffn_params")
        self.dim_ff = SPruneUnit(name, "dim_ff", dim_ff)
        self.dim_model = SPruneUnit(name, "dim_model", dim_model)
    
    def get_param_exp(self):
        return self.dim_ff.density * self.dim_model.density
    
    def get_param_all(self):
        return self.dim_ff.dim * self.dim_model.dim

    def is_training(self):
        return self.dim_ff.training | self.dim_model.training
    
    def set_pruning(self, root: torch.nn.Module):
        if self.is_training():
            for name, module in root.named_modules():
                if 'w_in.w' in name:
                    module.forward_unprune = module.forward
                    def prune_forward(module_self, *args, **kwargs):
                        out = module_self.forward_unprune(*args, **kwargs)
                        out = out * self.dim_ff.mask.to(out.device)
                        return out
                    module.forward = types.MethodType(prune_forward, module)
                elif 'w_out' in name:
                    module.forward_unprune = module.forward
                    def prune_forward(module_self, x, *args, **kwargs):
                        mask = self.dim_ff.mask
                        x = x * mask.to(x.device)
                        out = module_self.forward_unprune(x, *args, **kwargs)
                        out = out * self.dim_model.mask.to(out.device)
                        return out
                    module.forward = types.MethodType(prune_forward, module)

    def prune_params(self, name: str, tensor: torch.Tensor):
        dim_ff_indices = torch.nonzero(self.dim_ff.mask == 1., as_tuple=True)[0]
        new_dim_ff_dim, old_dim_ff_dim = dim_ff_indices.size(0), self.dim_ff.dim
        dim_model_indices = torch.nonzero(self.dim_model.mask == 1., as_tuple=True)[0]
        new_dim_model_dim, old_dim_model_dim = dim_model_indices.size(0), self.dim_model.dim
        if 'w_out.weight' not in name:
            old_tensor = tensor.view(old_dim_ff_dim, old_dim_model_dim)
            new_tensor = old_tensor[dim_ff_indices, :][:, dim_model_indices]
            new_tensor = new_tensor.view(new_dim_ff_dim, new_dim_model_dim)
        else:
            old_tensor = tensor.permute(1, 0).contiguous().view(old_dim_ff_dim, old_dim_model_dim)
            new_tensor = old_tensor[dim_ff_indices, :][:, dim_model_indices]
            new_tensor = new_tensor.view(new_dim_ff_dim, new_dim_model_dim).permute(1, 0).contiguous()
        return new_tensor


class AttUnit(SPruneUnit):
    def __init__(self, num_heads: int, dim_head: int, dim_model: int, mat_num: int, name: str):
        super().__init__(name, list_in="att")
        self.n_head_unit = HeadParamUnit(num_heads, dim_head, dim_model, self.name+'.self_attention')
        
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.mat_num = mat_num

    def is_training(self):
        return self.training | self.n_head_unit.is_training()

    def get_param_exp(self):
        if self.is_training():
            ret = self.density * (self.n_head_unit.get_param_exp() * self.mat_num + self.dim_model)
        else:
            ret = 0
        return ret

    def get_param_all(self):
        if self.is_training():
            ret = self.n_head_unit.get_param_all() * self.mat_num + self.dim_model
        else:
            ret = 0
        return  ret

    def set_pruning(self, module: torch.nn.Module):
        if self.training:
            module.forward_unprune = module.forward
            def prune_forward(module_self, hidden_states, *args, **kwargs):
                mask = self.mask.to(hidden_states.device)
                out = module_self.forward_unprune(hidden_states, *args, **kwargs)
                out = (hidden_states.to(torch.float32) + (out.to(torch.float32) - hidden_states.to(torch.float32)) * mask.to(torch.float32)).to(torch.half)
                return out
            module.forward = types.MethodType(prune_forward, module)


class FFNUnit(SPruneUnit):
    def __init__(self, dim_ff: int, dim_model: int, mat_num: int, name: str):
        super().__init__(name, list_in="ffn")
        self.dim_model = dim_model
        self.param = FFParamUnit(dim_ff, dim_model, self.name+".ffn")
        self.param_num = mat_num

    def is_training(self):
        return self.training | self.param.is_training()

    def get_param_exp(self):
        if self.is_training():
            ret = self.density * (self.param.get_param_exp() * self.param_num + self.dim_model)
        else:
            ret = 0
        return ret

    def get_param_all(self):
        if self.is_training():
            ret = self.param.get_param_all() * self.param_num + self.dim_model
        else:
            ret = 0
        return  ret

    def set_pruning(self, module: torch.nn.Module):
        if self.training:
            module.forward_unprune = module.forward
            def prune_forward(module_self, hidden_states, *args, **kwargs):
                mask = self.mask.to(hidden_states.device)
                out = module_self.forward_unprune(hidden_states, *args, **kwargs)
                out = (hidden_states.to(torch.float32) + (out.to(torch.float32) - hidden_states.to(torch.float32)) * mask.to(torch.float32)).to(torch.half)
                return out
            module.forward = types.MethodType(prune_forward, module)


class TransformerUnit(SPruneUnit):
    def __init__(
        self, 
        module: torch.nn.Module, 
        name: str,
        ):
        super().__init__(name, list_in="transformer")

        if hasattr(module, 'self_att'):
            num_heads = module.self_att.self_attention.num_heads
            dim_head = module.self_att.self_attention.dim_head
            dim_model = module.self_att.self_attention.dim_model    
            self.att = AttUnit(num_heads, dim_head, dim_model, mat_num=4, name=self.name+'.self_att')
        elif hasattr(module, 'cross_att'):
            num_heads = module.cross_att.self_attention.num_heads
            dim_head = module.cross_att.self_attention.dim_head
            dim_model = module.cross.self_attention.dim_model    
            self.att = AttUnit(num_heads, dim_head, dim_model, mat_num=4, name=self.name+'.cross_att')

        if hasattr(module, 'ffn'):
            dim_model = get_dim_model(module.ffn.ffn)
            dim_ff = get_dim_ff(module.ffn.ffn)
            if hasattr(module.ffn.ffn.w_in, 'w_1'):
                mat_num = 3
            else:
                mat_num = 2
            self.ffn = FFNUnit(dim_ff, dim_model, mat_num, name=self.name+'.ffn')

    def get_param_exp(self):
        att_param = self.att.get_param_exp() if hasattr(self, "att") else 0.
        ffn_param = self.ffn.get_param_exp() if hasattr(self, "ffn") else 0.
        res = self.density * (att_param + ffn_param)
        return res

    def get_param_all(self):
        att_param = self.att.get_param_all() if hasattr(self, "att") else 0.
        ffn_param = self.ffn.get_param_all() if hasattr(self, "ffn") else 0.
        res = att_param + ffn_param
        return res

    def set_pruning(self, module: torch.nn.Module):
        if isinstance(module, bmt.CheckpointBlock):
            module.forward_unprune = module._module.forward
        else:
            module.forward_unprune = module.forward

        def prune_forward(module_self, self_hidden_states, *args, **kwargs):
            mask = self.mask.to(self_hidden_states.device)
            out = module_self.forward_unprune(self_hidden_states, *args, **kwargs)
            out = (self_hidden_states.to(torch.float32) + (out.to(torch.float32) - self_hidden_states.to(torch.float32)) * mask.to(torch.float32)).to(torch.half)
            # out = (self_hidden_states.to(torch.float32) + (out.to(torch.float32) - self_hidden_states.to(torch.float32)) * 0.1).to(torch.half)
            return out

        if isinstance(module, bmt.CheckpointBlock):
            module._module.forward = types.MethodType(prune_forward, module)
        else:
            module.forward = types.MethodType(prune_forward, module)
