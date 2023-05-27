import inspect
import functools
import bmtrain as bmt
from typing import Optional, List
from collections import OrderedDict
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from model_center.model import ModelOutput

from .distilling import BMDistill
from .pruning import BMPrune
from .moe import BMMoE
from .quant import BMQuant
from .utils.config import ConfigParser

class CookOutput:
    def __init__(self, 
        loss, 
        original_output, 
        d_loss = 0., 
        sprune_plugin = None, 
        sprune_engine = None
        ):

        self.loss = loss
        self.original_output = original_output
        self.d_loss = d_loss
        self.sprune_plugin = sprune_plugin
        self.sprune_engine = sprune_engine


class CookTrainer:
    r"""A basic training manager of BMCook.

    In BMCook, the distillation and pruning will introduce distill_loss, lagrangian_loss and sparsity. 
    They are necessary for gradient backpropagation. CookTrainer will combine the original model outputs
    and these necessary variables and return to users together.

    Example::
    >>> # setup the forward
    >>> CookTrainer.set_forward(your_cook_config, your_model, your_optimizer, your_teacher_model)
    >>> ...
    >>> # use the forward
    >>> outputs = CookTrainer.forward(your_model, your_inputs, your_loss_func)
    """

    @staticmethod
    def forward(model: Module, loss_func: Module, targets: Tensor, model_args: List, model_kwargs: OrderedDict) -> List:
        r"""The core method in :class:`CookTrainer`.

        it combines the original model.forward and loss calculation.

        it should be rewrite in :method:`CookTrainer.set_forward`, if not an AttributeError will be raised.

        Args:
            model: target compressing model, basically from ModelCenter PLMs.
            loss_func: the loss function. supports the format: loss_func(input: Tensor, target: Tensor) -> Tensor
            targets: the target used to calculate loss.
            model_args: args of :method:`model.forward`
            model_kwargs: kwargs of :method:`model.forward`

        Return:
            `[loss, model_outputs, lag_loss, sparsity, d_loss, moe_hidden]`
        """
        raise NotImplementedError("The staticmethod forward() should be defined in :method:`set_forward`.")

    @classmethod
    def set_compression(
        cls,
        cook_config: ConfigParser,
        model: Optional[Module] = None,
        optimizer: Optional[Optimizer] = None,
        teacher: Optional[Module] = None,
        quant_layer_cls = None
    ):
        r"""Define the :method:`forward`, and set up :class:`BMPrune`, :class:`BMDistill`, :class:`BMQuant`
        and :class:`BMMoE`.

        :class:`BMPrune` should be ahead of :class:`BMDistill`, because the :method:`forward` is 
        changed in both :class:`BMPrune` and :class:`BMDistill`.

        The output format: `[loss, logits, lag_loss, sparsity, distill_loss]`

        Args:
            cook_config: BMCook config. You can set the compression strategy in this config and use 
                `bmcook.utils.config.ConfigParse` to parse it.
            model: target compressing model, basically from ModelCenter PLMs.
            optimizer: optimizer used to train model.
            teacher: teacher model used to distillation, basically from ModelCenter PLMs.
        """
        assert model is not None, "model is necessary for any compression setup."
        
        model_args_list = inspect.getfullargspec(model.forward).args
        if teacher is not None:
            teacher_args_list = inspect.getfullargspec(teacher.forward).args
            if teacher_args_list != model_args_list:
                raise ValueError("the techer forward func differs from model forward func.")
        
        if 'return_logits' in model_args_list:
            model.forward = functools.partial(model.forward, return_logits=True)
        elif 'output_logits' in model_args_list:
            model.forward = functools.partial(model.forward, output_logits=True)

        def forward(model, loss_func, targets, *model_args, **model_kwargs):
            outputs = model(
                *model_args, **model_kwargs
            )
            if isinstance(outputs, ModelOutput):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                raise TypeError(f"model output should be model_center.model.ModelOutput or tuple, but got {type(outputs)}")

            loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))

            ret = CookOutput(loss, outputs)
            return ret


        forward_doc = cls.forward.__doc__
        cls.forward = forward
        cls.forward.__doc__ = forward_doc

        # for pruning
        BMPrune.compute_mask(model, cook_config)
        BMPrune.set_optim_for_pruning(optimizer)

        # for distillation
        cls.forward = BMDistill.set_forward(model, teacher, cls.forward, cook_config)

        # for quantization
        BMQuant.quantize(model, cook_config, quant_layer_cls)

        # for moefication
        cls.forward = BMMoE.get_hidden(model, cook_config, cls.forward)

        bmt.synchronize()
