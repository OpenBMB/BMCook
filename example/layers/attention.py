from typing import Optional
import torch
import bmpretrain as bmp
import cpm_kernels.torch as ct
import math

class Attention(bmp.DistributedModule):
    def __init__(self,
            dim_model : int,
            num_heads : int,
            dim_head : int,
            init_method : bmp.ParameterInitializer,
            int8=True,
            dtype=torch.half
        ):
        super().__init__()

        self.project_q = bmp.DistributedParameter(
            torch.empty(num_heads * dim_head, dim_model, dtype=dtype),
            init_method=init_method)
        self.project_k = bmp.DistributedParameter(
            torch.empty(num_heads * dim_head, dim_model, dtype=dtype),
            init_method=init_method)
        self.project_v = bmp.DistributedParameter(
            torch.empty(num_heads * dim_head, dim_model, dtype=dtype),
            init_method=init_method)

        self.attention_out = bmp.DistributedParameter(
            torch.empty(dim_model, num_heads * dim_head, dtype=dtype),
            init_method=init_method)

        self.relu = torch.nn.ReLU()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.int8 = int8

    def forward(self, 
            hidden_q : torch.Tensor,                # (batch, dim_model, len_q)
            hidden_kv : torch.Tensor,               # (batch, dim_model, len_k)
            mask : torch.Tensor,                    # (batch, len_k, len_q)
            position_bias : Optional[torch.Tensor], # (num_heads, len_k, len_q)
        ):
        """
        Args:
            hidden_q : (batch, dim_model, len_q)    fp16
            hidden_kv : (batch, dim_model, len_k)   fp16
            mask : (batch, len_k, len_q)            fp16
            position_bias : (num_heads, len_k, len_q)   fp16
        Returns:
            out : (batch, dim_model, len_q)         fp16
        """

        # bmp.inspect.record_tensor(hidden_q, "attn_x")

        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(2)
        len_k = hidden_kv.size(2)

        project_q = self.project_q
        project_k = self.project_k
        project_v = self.project_v

        attention_out = self.attention_out

        # (1#batch, num_heads * dim_head, dim_model) @ (batch, dim_model, len_q) = (batch, num_heads * dim_head, len_q)
        h_q = ct.bmm(project_q.unsqueeze(0), False, hidden_q, False, int8=self.int8) #/ math.sqrt(self.dim_model)
        h_k = ct.bmm(project_k.unsqueeze(0), False, hidden_kv, False, int8=self.int8) #/ math.sqrt(self.dim_model)
        h_v = ct.bmm(project_v.unsqueeze(0), False, hidden_kv, False, int8=self.int8) #/ math.sqrt(self.dim_model)

        # view (batch * num_heads, dim_head, length)
        h_q = h_q.view(batch_size * self.num_heads, self.dim_head, -1)
        h_k = h_k.view(batch_size * self.num_heads, self.dim_head, -1)
        h_v = h_v.view(batch_size * self.num_heads, self.dim_head, -1)

        # (batch * num_heads, dim_head, len_k)T @ (batch * num_heads, dim_head, len_q) = (batch * num_heads, len_k, len_q)
        score = ct.bmm( h_k, True, h_q, False, int8=False)  # use FP 16 here
        score = score / math.sqrt(self.dim_head)
        
        # (batch, num_heads, len_k, len_q) 
        score = score.view(batch_size, self.num_heads, len_k, len_q)
        if position_bias is not None:
            score = ct.batched_add(
                score,   
                position_bias
            )
        
        # (batch, num_heads, len_k * len_q)
        masked_score = ct.mask(
            score.view(batch_size, self.num_heads, -1),
            mask.view(batch_size, -1),
            float("-inf")
        )

        # (batch * num_heads, len_k, len_q)
        masked_score = masked_score.view(batch_size * self.num_heads, len_k, len_q)

        # (batch * num_heads, len_k, len_q)
        masked_score = ct.softmax(masked_score) # softmax along len_k

        # (batch * num_heads, dim_head, len_k) @ (batch * num_heads, len_k, len_q) = (batch * num_heads, dim_head, len_q)
        attention_result = ct.bmm(h_v, False, masked_score, False, int8=False)  # use FP 16 here

        attention_result = attention_result.view(batch_size, self.num_heads * self.dim_head, len_q)
        # (1#batch, dim_model, num_heads * dim_head) @ (batch, num_heads * dim_head, len_q) = (batch, dim_model, len_q)
        attention_out = ct.bmm(attention_out.unsqueeze(0), False, attention_result, False, int8=self.int8) #/ math.sqrt(self.dim_head * self.num_heads)
        return attention_out

def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-2]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("j , i -> i j", torch.arange(seq_len), inv_freq).to(x.device).half()
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, ::2, :]
    x2 = x[:, 1::2, :]
    x = torch.stack((-x2, x1), axis=-2)
    return x.flatten(-3, -2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: t[None, :, offset : x.shape[-1] + offset].repeat_interleave(2, 1), sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)

class GPTJAtt(bmp.DistributedModule):
    def __init__(self,
            dim_model : int,
            num_heads : int,
            dim_head : int,
            init_method: bmp.ParameterInitializer,
            int8=True,
            dtype=torch.half
        ):
        super().__init__()

        self.q_proj = bmp.DistributedParameter(
            torch.empty(num_heads * dim_head, dim_model, dtype=dtype),
            init_method=init_method)
        self.k_proj = bmp.DistributedParameter(
            torch.empty(num_heads * dim_head, dim_model, dtype=dtype),
            init_method=init_method)
        self.v_proj = bmp.DistributedParameter(
            torch.empty(num_heads * dim_head, dim_model, dtype=dtype),
            init_method=init_method)

        self.out_proj = bmp.DistributedParameter(
            torch.empty(dim_model, num_heads * dim_head, dtype=dtype), 
            init_method=init_method)

        self.relu = torch.nn.ReLU()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.int8 = int8

        self.rotary_dim = 64

    def forward(self, 
            hidden_q : torch.Tensor,                # (batch, dim_model, len_q)
            hidden_kv : torch.Tensor,               # (batch, dim_model, len_k)
            mask : torch.Tensor,                    # (batch, len_k, len_q)
            position_bias : Optional[torch.Tensor], # (num_heads, len_k, len_q)
        ):
        """
        Args:
            hidden_q : (batch, dim_model, len_q)    fp16
            hidden_kv : (batch, dim_model, len_k)   fp16
            mask : (batch, len_k, len_q)            fp16
            position_bias : (num_heads, len_k, len_q)   fp16
        Returns:
            out : (batch, dim_model, len_q)         fp16
        """

        # bmp.inspect.record_tensor(hidden_q, "attn_x")

        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(2)
        len_k = hidden_kv.size(2)

        project_q = self.q_proj
        project_k = self.k_proj
        project_v = self.v_proj

        attention_out = self.out_proj

        # (1#batch, num_heads * dim_head, dim_model) @ (batch, dim_model, len_q) = (batch, num_heads * dim_head, len_q)
        h_q = ct.bmm(project_q.unsqueeze(0), False, hidden_q, False, int8=self.int8) #/ math.sqrt(self.dim_model)
        h_k = ct.bmm(project_k.unsqueeze(0), False, hidden_kv, False, int8=self.int8) #/ math.sqrt(self.dim_model)
        h_v = ct.bmm(project_v.unsqueeze(0), False, hidden_kv, False, int8=self.int8) #/ math.sqrt(self.dim_model)

        # view (batch * num_heads, dim_head, length)
        h_q = h_q.view(batch_size * self.num_heads, self.dim_head, -1)
        h_k = h_k.view(batch_size * self.num_heads, self.dim_head, -1)
        h_v = h_v.view(batch_size * self.num_heads, self.dim_head, -1)

        k_rot = h_k[:, : self.rotary_dim, :]
        k_pass = h_k[:, self.rotary_dim :, :]

        q_rot = h_q[:, : self.rotary_dim, :]
        q_pass = h_q[:, self.rotary_dim :, :]

        seq_len = h_k.shape[-1]
        sincos = fixed_pos_embedding(k_rot, -1, seq_len=seq_len)
        k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=0)
        q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=0)

        h_k = torch.cat([k_rot, k_pass], dim=-2)
        h_q = torch.cat([q_rot, q_pass], dim=-2)

        # (batch * num_heads, dim_head, len_k)T @ (batch * num_heads, dim_head, len_q) = (batch * num_heads, len_k, len_q)
        score = ct.bmm( h_k, True, h_q, False, int8=False)  # use FP 16 here
        score = score / math.sqrt(self.dim_head)
        
        # (batch, num_heads, len_k, len_q) 
        score = score.view(batch_size, self.num_heads, len_k, len_q)
        # if position_bias is not None:
        #     score = ct.batched_add(
        #         score,   
        #         position_bias
        #     )
        
        # (batch, num_heads, len_k * len_q)
        masked_score = ct.mask(
            score.view(batch_size, self.num_heads, -1),
            mask.view(batch_size, -1),
            float("-inf")
        )

        # (batch * num_heads, len_k, len_q)
        masked_score = masked_score.view(batch_size * self.num_heads, len_k, len_q)

        self.masked_score = masked_score  # Intermediary values for KD

        # (batch * num_heads, len_k, len_q)
        masked_score = ct.softmax(masked_score) # softmax along len_k

        # (batch * num_heads, dim_head, len_k) @ (batch * num_heads, len_k, len_q) = (batch * num_heads, dim_head, len_q)
        attention_result = ct.bmm(h_v, False, masked_score, False, int8=False)  # use FP 16 here

        attention_result = attention_result.view(batch_size, self.num_heads * self.dim_head, len_q)
        # (1#batch, dim_model, num_heads * dim_head) @ (batch, num_heads * dim_head, len_q) = (batch, dim_model, len_q)
        attention_out = ct.bmm(attention_out.unsqueeze(0), False, attention_result, False, int8=self.int8) #/ math.sqrt(self.dim_head * self.num_heads)
        return attention_out

