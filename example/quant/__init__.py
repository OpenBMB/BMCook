
class BMQuant:
    '''
    BMQuant enables quantization-aware training of PLMs.

    To use this module, you need to implement the linear transformations with the following format:

    ```python
    import cpm_kernels.torch as ct
    ct.bmm(w.unsqueeze(0), False, x, False, int8=self.int8)
    ```

    where `w` is the weight tensor, `x` is the input tensor. Modules have the attribute `int8`, which is a boolean, to control whether the operation is quantized.
    '''

    @classmethod
    def quantize(cls, model):
        '''
        Recursively change the linear transformations in the model to quantized version. Current implementation only supports GPT-J. To customize the quantization for other models, you can override this method. In the future, we will support customization by additional configuration files instead of hard coding.

        :param model: Model to quantize.
        '''
        # Will implement more generally in the next version to support arbitrary models
        for layer_idx in range(len(model.dec_layers)):
            layer = model.dec_layers[layer_idx]
            layer._module.ff.int8 = True
            layer._module.self_attention.int8 = True
