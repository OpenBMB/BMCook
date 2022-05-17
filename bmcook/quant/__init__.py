import types
import model_center
import math
import cpm_kernels.torch as ct

class BMQuant:
    '''
    BMQuant enables quantization-aware training of PLMs.

    To use this module, you need to implement the linear transformations with the following format:

    `import cpm_kernels.torch as ct`
    
    `ct.bmm(w.unsqueeze(0), False, x, False, int8=self.int8)`

    where `w` is the weight tensor, `x` is the input tensor. Modules have the attribute `int8`, which is a boolean, to control whether the operation is quantized.
    '''

    @classmethod
    def quantize(cls, model, config):
        '''
        Recursively change the linear transformations in the model to quantized version. Current implementation only supports GPT-J. To customize the quantization for other models, you can override this method. In the future, we will support customization by additional configuration files instead of hard coding.

        :param model: Model to quantize.
        '''

        # fix cpm_kernel
        ct.gemm.GEMMInt8._backward = ct.gemm.GEMMInt8.backward
        def new_func(ctx, grad_f):
            if not grad_f.is_contiguous():
                grad_f = grad_f.contiguous()
            return ct.gemm.GEMMInt8._backward(ctx, grad_f)
        ct.gemm.GEMMInt8.backward = new_func

        quant_config = config.get('quantization')
        if not quant_config['is_quant']:
            return

        for name, module in model.named_modules():
            if isinstance(module, model_center.layer.Linear):
                if len(quant_config["quantized_module"]) != 0:
                    if not any([pattern in name for pattern in quant_config["quantized_module"]]):
                        continue
                module.forward = types.MethodType(forward_in8, module)

def forward_in8(module_self, x):
    if module_self.length_scale and module_self.length_scale_before:
        x = x / math.sqrt(module_self.dim_in)
    x = x.transpose(1, 2).contiguous()
    x = ct.bmm(module_self.weight.unsqueeze(0), False, x, False, int8=True)
    x = x.transpose(1, 2).contiguous()
    if module_self.length_scale and not module_self.length_scale_before:
        x = x / math.sqrt(module_self.dim_in)
    if module_self.bias is not None:
        x = x + module_self.bias
    return x