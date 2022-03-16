import torch
import numpy as np
import bmtrain as bmp
from typing import List

import layers


class T5(torch.nn.Module):
    def __init__(self, 
            num_enc : int, num_dec : int,                                       # layers
            dim_model : int, num_heads : int, dim_head : int, dim_ff : int,     # shapes
            vocab_input_size : int, vocab_output_size : int,                    # inputs
            position_bias_num_buckets : int, position_bias_max_distance : int,
            eps : float = 1e-6, int8 : bool = True, dtype : torch.dtype = torch.half
        ):
        super().__init__()
        
        self.num_enc = num_enc
        self.num_dec = num_dec

        self.enc_layers = bmp.TransformerBlockList([
            bmp.CheckpointBlock(
                layers.TransformerEncoder(dim_model, num_heads, dim_head, dim_ff, eps, int8=int8, dtype=dtype)
            )
            for _ in range(num_enc)
        ])

        self.dec_layers = bmp.TransformerBlockList([
            bmp.CheckpointBlock(
                layers.TransformerDecoder(dim_model, num_heads, dim_head, dim_ff, eps, int8=int8, dtype=dtype)
            )
            for _ in range(num_dec)
            
        ])

        self.layernorm_after_enc = layers.LayerNorm(dim_model, eps, bias=False, dtype=dtype)
        self.layernorm_after_dec = layers.LayerNorm(dim_model, eps, bias=False, dtype=dtype)

        self.input_embedding = layers.Embedding(vocab_input_size, dim_model, dtype=dtype)
        self.output_projection = layers.Projection(vocab_output_size, dim_model, dtype=dtype)

        self.position_bias_enc = layers.PositionEmbedding(num_heads, position_bias_num_buckets, position_bias_max_distance, bidirectional=True, dtype=dtype)
        self.position_bias_dec = layers.PositionEmbedding(num_heads, position_bias_num_buckets, position_bias_max_distance, bidirectional=False, dtype=dtype)
    

    def forward(self, 
            enc_input : torch.Tensor,       # (batch, seq_enc),
            enc_length : torch.Tensor,      # (batch),

            dec_input : torch.Tensor,       # (batch, seq_dec),
            dec_length : torch.Tensor,      # (batch),
        ):
        """
        Args:
            enc_input: (batch, seq_enc)     int32
            enc_length: (batch)             int32
            dec_input: (batch, seq_dec)     int32
            dec_length: (batch)             int32
        Returns:
            logits : (batch, seq_dec, vocab_output_size)
        """
        batch = enc_input.size(0)
        seq_enc = enc_input.size(1)
        seq_dec = dec_input.size(1)

        device = enc_input.device

        enc_mask_1d = torch.arange(seq_enc, device=device)[None, :].repeat(batch, 1) < enc_length[:, None]
        dec_mask_1d = torch.arange(seq_dec, device=device)[None, :].repeat(batch, 1) < dec_length[:, None]
        directional_mask = torch.arange(seq_dec, device=device).view(-1, 1) <= torch.arange(seq_dec, device=device)

        # (batch, seq_enc, seq_enc)
        enc_mask = enc_mask_1d.view(batch, seq_enc, 1) & enc_mask_1d.view(batch, 1, seq_enc)

        # (batch, seq_dec, seq_dec)
        dec_mask = dec_mask_1d.view(batch, seq_dec, 1) & dec_mask_1d.view(batch, 1, seq_dec) & directional_mask.view(1, seq_dec, seq_dec)

        # (batch, seq_enc, seq_dex)
        cross_mask = enc_mask_1d.view(batch, seq_enc, 1) & dec_mask_1d.view(batch, 1, seq_dec)

        # (num_heads, seq_enc, seq_enc)
        position_bias_enc = self.position_bias_enc(seq_enc, seq_enc)

        # (num_heads, seq_dec, seq_dec)
        position_bias_dec = self.position_bias_dec(seq_dec, seq_dec)

        # (batch, dim_model, seq_enc)
        hidden_enc = self.input_embedding(enc_input)

        hidden_enc = self.enc_layers(hidden_enc, enc_mask, position_bias_enc)    
            
        hidden_enc = self.layernorm_after_enc(hidden_enc)

        hidden_dec = self.input_embedding(dec_input)
        
        hidden_dec = self.dec_layers(hidden_dec, hidden_enc, dec_mask, cross_mask, position_bias_dec, None)
 
        # (batch, dim_model, seq_dec)
        hidden_dec = self.layernorm_after_dec(hidden_dec)
        logits = self.output_projection(hidden_dec)
        return logits

class GPT(torch.nn.Module):
    def __init__(self, 
            num_dec : int,                                                      # layers
            dim_model : int, num_heads : int, dim_head : int, dim_ff : int,     # shapes
            vocab_size : int,                                                   # inputs
            init_std : float,
            position_bias_num_buckets : int, position_bias_max_distance : int,
            eps : float = 1e-6, int8 : bool = True, dtype : torch.dtype = torch.half
        ):
        super().__init__()
        
        init_method = bmp.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=init_std)

        self.num_dec = num_dec

        self.dec_layers = bmp.TransformerBlockList([
            bmp.CheckpointBlock(
                layers.TransformerDecoder(dim_model, num_heads, dim_head, 
                    dim_ff, eps, init_method=init_method, int8=int8,
                    dtype=dtype, cross_attn=False
                )
            )
            for _ in range(num_dec)
        ])

        self.layernorm_after_dec = layers.LayerNorm(dim_model, eps, bias=False, 
            dtype=dtype)

        self.input_embedding = layers.Embedding(vocab_size, dim_model,
            init_method=init_method, dtype=dtype)

        self.position_bias_dec = layers.PositionEmbedding(num_heads, 
            position_bias_num_buckets, position_bias_max_distance, 
            init_method=init_method, bidirectional=False, dtype=dtype)
    

    def forward(self, 
            dec_input : torch.Tensor,       # (batch, seq_dec),
            dec_length : torch.Tensor,      # (batch),
            output_hidden_states : bool=False,
        ):
        """
        Args:
            dec_input: (batch, seq_dec)     int32
            dec_length: (batch)             int32
        Returns:
            logits : (batch, seq_dec, vocab_output_size)
        """
        batch = dec_input.size(0)
        seq_dec = dec_input.size(1)

        device = dec_input.device

        dec_mask_1d = torch.arange(seq_dec, device=device)[None, :].repeat(batch, 1) < dec_length[:, None]
        directional_mask = torch.arange(seq_dec, device=device).view(-1, 1) <= torch.arange(seq_dec, device=device)

        # (batch, seq_dec, seq_dec)
        dec_mask = dec_mask_1d.view(batch, seq_dec, 1) & dec_mask_1d.view(batch, 1, seq_dec) & directional_mask.view(1, seq_dec, seq_dec)

        # (num_heads, seq_dec, seq_dec)
        position_bias_dec = self.position_bias_dec(seq_dec, seq_dec)

        # (batch, dim_model, seq_enc)
        hidden_dec = self.input_embedding(dec_input)
        
        hidden_dec = self.dec_layers(hidden_dec, None, dec_mask, None, position_bias_dec, None)
 
        # (batch, dim_model, seq_dec)
        hidden_dec = self.layernorm_after_dec(hidden_dec)
        logits = self.input_embedding.proj(hidden_dec)
        if output_hidden_states:
            return logits, hidden_dec
        return (logits, )

    def get_sparsity(self) -> List[float]:
        sparsity = []
        for ckpt_block in self.dec_layers._modules.values():
            decoder = ckpt_block._module
            sparsity.append(decoder.get_sparsity())
        return sparsity
    
    def reset_sparsity(self):
        for ckpt_block in self.dec_layers._modules.values():
            decoder = ckpt_block._module
            decoder.reset_sparsity()

    def stop_recording_relu_distr(self) -> List[np.ndarray]:
        relu_distr = []
        for ckpt_block in self.dec_layers._modules.values():
            decoder = ckpt_block._module
            relu_distr.append(decoder.get_relu_distr())
            decoder.stop_recording_relu_distr()
        return relu_distr

    def start_recording_relu_distr(self):
        for ckpt_block in self.dec_layers._modules.values():
            decoder = ckpt_block._module
            decoder.start_recording_relu_distr()

class GPTJ(torch.nn.Module):
    def __init__(self, 
            num_dec : int,                                                      # layers
            dim_model : int, num_heads : int, dim_head : int, dim_ff : int,     # shapes
            vocab_size : int,                                                   # inputs
            init_std : float,
            position_bias_num_buckets : int, position_bias_max_distance : int,
            eps : float = 1e-5, int8 : bool = True, dtype : torch.dtype = torch.half,
            act_func='gelu'
        ):
        super().__init__()

        self.num_dec = num_dec
        self.dim_model = dim_model

        init_method = bmp.ParameterInitializer(torch.nn.init.normal_, mean=0.0, std=init_std)

        self.dec_layers = bmp.TransformerBlockList([
            bmp.CheckpointBlock(
                layers.GPTJDecoder(dim_model, num_heads, dim_head, dim_ff, eps, int8=int8, dtype=dtype, cross_attn=False, init_method=init_method, act_func=act_func)
            )
            for _ in range(num_dec)
        ])

        self.layernorm_after_dec = layers.LayerNorm(dim_model, eps, bias=True, dtype=dtype)

        self.input_embedding = layers.Embedding(vocab_size, dim_model, dtype=dtype, init_method=init_method)

        self.lm_head = layers.LMHead(vocab_size, dim_model, init_method=init_method, dtype=dtype)

    def forward(self, 
            dec_input : torch.Tensor,       # (batch, seq_dec),
            dec_length : torch.Tensor,      # (batch),
            output_hidden_states : bool=False,
        ):
        """
        Args:
            dec_input: (batch, seq_dec)     int32
            dec_length: (batch)             int32
        Returns:
            logits : (batch, seq_dec, vocab_output_size)
        """
        batch = dec_input.size(0)
        seq_dec = dec_input.size(1)

        device = dec_input.device

        dec_mask_1d = torch.arange(seq_dec, device=device)[None, :].repeat(batch, 1) < dec_length[:, None]
        directional_mask = torch.arange(seq_dec, device=device).view(-1, 1) <= torch.arange(seq_dec, device=device)

        # (batch, seq_dec, seq_dec)
        dec_mask = dec_mask_1d.view(batch, seq_dec, 1) & dec_mask_1d.view(batch, 1, seq_dec) & directional_mask.view(1, seq_dec, seq_dec)

        # (batch, dim_model, seq_enc)
        hidden_dec = self.input_embedding(dec_input)
        
        hidden_dec = self.dec_layers(hidden_dec, None, dec_mask, None, None, None)
 
        # (batch, dim_model, seq_dec)
        hidden_dec = self.layernorm_after_dec(hidden_dec)
        logits = self.lm_head(hidden_dec)

        # Intermediary values for KD
        all_hidden_states = []
        all_att_scores = []
        for decoder in self.iter_decoders():
            all_hidden_states.append(decoder.normed_input)
            all_att_scores.append(decoder.att_score)
        all_hidden_states = all_hidden_states[1:] + [hidden_dec]

        return (logits, all_hidden_states, all_att_scores)

    def iter_decoders(self):
        for ckpt_block in self.dec_layers._modules.values():
            yield ckpt_block._module
