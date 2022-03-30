
class BMQuant:
    '''
    BMQuant enables quantization-aware training of PLMs.
    '''

    @classmethod
    def quantize(cls, model):
        '''
        Recursively change the linear transformations in the model to quantized version.

        :param model: Model to quantize.
        '''
        # Will implement more generally in the next version to support arbitrary models
        for layer_idx in range(len(model.dec_layers)):
            layer = model.dec_layers[layer_idx]
            layer._module.ff.int8 = True
            layer._module.self_attention.int8 = True
