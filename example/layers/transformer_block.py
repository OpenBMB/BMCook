from typing import Optional
import torch
import cpm_kernels.torch as ct
import numpy as np

from .attention import Attention, GPTJAtt
from .layernorm import LayerNorm
from .feedforward import FeedForward, GPTJFF
import bmtrain as bmt

class TransformerEncoder(torch.nn.Module):
    def __init__(self,
            dim_model : int,
            num_heads : int,
            dim_head : int,
            dim_ff : int,
            eps : float,
            init_method : bmp.ParameterInitializer,
            int8=True,
            dtype=torch.half):
        super().__init__()

        self.layernorm_before_attention = LayerNorm(dim_model, eps, bias=False, dtype=dtype)
        self.self_attention = Attention(dim_model, num_heads, dim_head,
            init_method=init_method, int8=int8, dtype=dtype)

        self.layernorm_before_ff = LayerNorm(dim_model, eps, bias=False, dtype=dtype)
        self.ff = FeedForward(dim_model, dim_ff, int8=int8, dtype=dtype)

    def forward(self,
            hidden_state : torch.Tensor,    # (batch, hidden_size, seq_len)
            mask : torch.Tensor,            # (batch, seq_len, seq_len)
            position_bias : torch.Tensor,   # (num_heads, seq_len, seq_len)
        ):
        """
        Args:
            hidden_state: (batch, hidden_size, seq_len)     fp16
            mask: (batch, seq_len, seq_len)                 fp16
            position_bias: (num_heads, seq_len, seq_len)    fp16
        Returns:
            out: (batch, hidden_size, seq_len)              fp16
        """
        x = self.layernorm_before_attention(hidden_state)
        x = self.self_attention(x, x, mask, position_bias)
        hidden_state = ct.element_add(hidden_state, x)      # hidden_state = hidden_state + x


        x = self.layernorm_before_ff(hidden_state)
        x = self.ff(x)
        hidden_state = ct.element_add(hidden_state, x)      # hidden_state = hidden_state + x

        return hidden_state

class TransformerDecoder(torch.nn.Module):
    def __init__(self,
            dim_model : int,
            num_heads : int,
            dim_head : int,
            dim_ff : int,
            eps : float,
            init_method : bmp.ParameterInitializer,
            int8=True,
            dtype=torch.half,
            cross_attn=True):
        super().__init__()

        self.layernorm_before_self_attention = LayerNorm(dim_model, eps, bias=False, dtype=dtype)
        self.self_attention = Attention(dim_model, num_heads, dim_head,
            init_method=init_method, int8=int8, dtype=dtype)

        self.cross_attn = cross_attn
        if cross_attn:
            self.layernorm_before_cross_attention = LayerNorm(dim_model, eps, bias=False, dtype=dtype)
            self.cross_attention = Attention(dim_model, num_heads, dim_head, 
                init_method=init_method, int8=int8, dtype=dtype)

        self.layernorm_before_ff = LayerNorm(dim_model, eps, bias=False, dtype=dtype)
        self.ff = FeedForward(dim_model, dim_ff, init_method=init_method, int8=int8, dtype=dtype)

    def forward(self,
            hidden_state : torch.Tensor,                # (batch, hidden_size, seq_q)
            encoder_output : torch.Tensor,              # (batch, hidden_size, seq_k)
            mask_self_attn : torch.Tensor,              # (batch, seq_q, seq_q)
            mask_corss_attn : torch.Tensor,             # (batch, seq_k, seq_q)
            self_attn_bias : Optional[torch.Tensor],    # (num_heads, seq_q, seq_q)
            cross_attn_bias : Optional[torch.Tensor],   # (num_heads, seq_k, seq_q)
        ):
        """
        Args:
            hidden_state: (batch, hidden_size, seq_q)       fp16
            encoder_output: (batch, hidden_size, seq_k)     fp16
            mask_self_attn: (batch, seq_q, seq_q)           fp16
            mask_corss_attn: (batch, seq_k, seq_q)          fp16
            self_attn_bias: (num_heads, seq_q, seq_q)       fp16
            cross_attn_bias: (num_heads, seq_k, seq_q)      fp16
        Returns:
            out: (batch, hidden_size, seq_q)                fp16
        """
        x = self.layernorm_before_self_attention(hidden_state)
        x = self.self_attention(x, x, mask_self_attn, self_attn_bias)
        hidden_state = ct.element_add(hidden_state, x)

        if self.cross_attn:
            x = self.layernorm_before_cross_attention(hidden_state)
            x = self.cross_attention(x, encoder_output, mask_corss_attn, cross_attn_bias)
            hidden_state = ct.element_add(hidden_state, x)

        x = self.layernorm_before_ff(hidden_state)
        x = self.ff(x)
        hidden_state = ct.element_add(hidden_state, x)

        return hidden_state

    def reset_sparsity(self):
        self.ff.sparsity.reset()

    def get_sparsity(self) -> float:
        return self.ff.sparsity.get()

    def reset_relu_distr(self):
        self.ff.relu_distr.reset()
    
    def get_relu_distr(self) -> np.ndarray:
        return self.ff.relu_distr.get()
    
    def start_recording_relu_distr(self):
        self.reset_relu_distr()
        self.ff.record_relu_distr = True

    def stop_recording_relu_distr(self):
        self.ff.record_relu_distr = False


class GPTJDecoder(torch.nn.Module):
    def __init__(self, 
            dim_model : int, 
            num_heads : int, 
            dim_head : int, 
            dim_ff : int, 
            eps : float, 
            init_method : bmp.ParameterInitializer, 
            int8=True, 
            dtype=torch.half, 
            cross_attn=True,
            act_func='gelu'):
        super().__init__()

        self.ln_1 = LayerNorm(dim_model, eps, bias=True, dtype=dtype)
        self.self_attention = GPTJAtt(dim_model, num_heads, dim_head, init_method=init_method, int8=int8, dtype=dtype)
        self.ff = GPTJFF(dim_model, dim_ff, init_method=init_method, int8=int8, dtype=dtype, act_func=act_func)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self,
            hidden_state : torch.Tensor,                # (batch, hidden_size, seq_q)
            encoder_output : torch.Tensor,              # (batch, hidden_size, seq_k)
            mask_self_attn : torch.Tensor,              # (batch, seq_q, seq_q)
            mask_corss_attn : torch.Tensor,             # (batch, seq_k, seq_q)
            self_attn_bias : Optional[torch.Tensor],    # (num_heads, seq_q, seq_q)
            cross_attn_bias : Optional[torch.Tensor],   # (num_heads, seq_k, seq_q)
        ):
        """
        Args:
            hidden_state: (batch, hidden_size, seq_q)       fp16
            encoder_output: (batch, hidden_size, seq_k)     fp16
            mask_self_attn: (batch, seq_q, seq_q)           fp16
            mask_corss_attn: (batch, seq_k, seq_q)          fp16
            self_attn_bias: (num_heads, seq_q, seq_q)       fp16
            cross_attn_bias: (num_heads, seq_k, seq_q)      fp16
        Returns:
            out: (batch, hidden_size, seq_q)                fp16
        """
        x = self.ln_1(hidden_state)

        self.normed_input = x  # Intermediary values for KD

        x_1 = self.dropout(self.self_attention(x, x, mask_self_attn, self_attn_bias))

        self.att_score = self.self_attention.masked_score # Intermediary values for KD

        x_2 = self.dropout(self.ff(x))
        hidden_state = ct.element_add(hidden_state, x_1)
        hidden_state = ct.element_add(hidden_state, x_2)

        return hidden_state
