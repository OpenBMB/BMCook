
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
        model.int8 = True
