import torch
import random
import bmpretrain as bmp
import layers
from tqdm import tqdm
import time
from data import MMapIndexedDataset, Dataset
import numpy as np
from pruning import BMPrune, m4n2_2d_greedy, m4n2_2d_best
from distilling import init_student, set_forward

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
            position_bias_num_buckets : int, position_bias_max_distance : int,
            eps : float = 1e-6, int8 : bool = True, dtype : torch.dtype = torch.half
        ):
        super().__init__()
        
        self.num_dec = num_dec

        self.dec_layers = bmp.TransformerBlockList([
            bmp.CheckpointBlock(
                layers.TransformerDecoder(dim_model, num_heads, dim_head, dim_ff, eps, int8=int8, dtype=dtype, cross_attn=False)
            )
            for _ in range(num_dec)
            
        ])

        self.layernorm_after_dec = layers.LayerNorm(dim_model, eps, bias=False, dtype=dtype)

        self.input_embedding = layers.Embedding(vocab_size, dim_model, dtype=dtype)

        self.position_bias_dec = layers.PositionEmbedding(num_heads, position_bias_num_buckets, position_bias_max_distance, bidirectional=False, dtype=dtype)
    

    def forward(self, 
            dec_input : torch.Tensor,       # (batch, seq_dec),
            dec_length : torch.Tensor,      # (batch),
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
        return logits

class GPTJ(torch.nn.Module):
    def __init__(self, 
            num_dec : int,                                                      # layers
            dim_model : int, num_heads : int, dim_head : int, dim_ff : int,     # shapes
            vocab_size : int,                                                   # inputs
            position_bias_num_buckets : int, position_bias_max_distance : int,
            eps : float = 1e-5, int8 : bool = True, dtype : torch.dtype = torch.half
        ):
        super().__init__()
        
        self.num_dec = num_dec

        self.dec_layers = bmp.TransformerBlockList([
            bmp.CheckpointBlock(
                layers.GPTJDecoder(dim_model, num_heads, dim_head, dim_ff, eps, int8=int8, dtype=dtype, cross_attn=False)
            )
            for _ in range(num_dec)
            
        ])

        self.layernorm_after_dec = layers.LayerNorm(dim_model, eps, bias=True, dtype=dtype)

        self.input_embedding = layers.Embedding(vocab_size, dim_model, dtype=dtype)

        self.lm_head = layers.LMHead(vocab_size, dim_model, dtype=dtype)    

    def forward(self, 
            dec_input : torch.Tensor,       # (batch, seq_dec),
            dec_length : torch.Tensor,      # (batch),
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
        #bmp.print_rank(hidden_dec[0, :, 0])
        #exit()
        
        hidden_dec = self.dec_layers(hidden_dec, None, dec_mask, None, None, None)
 
        # (batch, dim_model, seq_dec)
        hidden_dec = self.layernorm_after_dec(hidden_dec)
        logits = self.lm_head(hidden_dec)
        return logits

def print_inspect(model, name):
    bmp.print_rank(
        bmp.inspect.format_summary(
            bmp.inspect.inspect_model(model, name)
        )
    )

class Trainer:

    @staticmethod
    def batch_iter(dataset : Dataset, batch_size, rank, world_size):
        st = 0
        end = len(dataset)
        batch = []
        while st < end:
            it = dataset[st + rank]
            if it is not None:
                batch.append( it )
            st += world_size
            if len(batch) == batch_size:
                yield {
                    "ctx": torch.stack([it["ctx"] for it in batch]),
                    "len_ctx": torch.LongTensor([it["len_ctx"] for it in batch]),
                    'target': torch.stack([it["target"] for it in batch]),
                }
                batch = []

    @staticmethod
    def forward(model, dec_input, dec_length, targets, loss_func):
        logits = model(dec_input, dec_length)
        batch, seq_len, vocab_out_size = logits.size()

        loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))

        return loss, logits

def main():
    bmp.init_distributed()

    #teacher = GPTJ(
    model = GPTJ(
        num_dec=28,
        dim_model=4096, num_heads=16, dim_head=256, dim_ff=16384,
        vocab_size=50400,
        position_bias_num_buckets=32, position_bias_max_distance=128,
        eps=1e-6, int8=True, dtype=torch.half
    )

    #model = GPTJ(
    #    num_dec=14,
    #    dim_model=4096, num_heads=16, dim_head=256, dim_ff=16384,
    #    vocab_size=50400,
    #    position_bias_num_buckets=32, position_bias_max_distance=128,
    #    eps=1e-6, int8=False, dtype=torch.half
    #)

    bmp.init_parameters(model)

    # for eval
    #bmp.load(model, "/data/home/scv0540/zzy/gpt-j/bm-gpt-int8.pt")
    bmp.load(model, "results-int8/ckpt-8000.pt")

    # for knowledge distillation
    #bmp.init_parameters(teacher)
    #print_inspect(model, "*")
    #bmp.load(teacher, "/data/home/scv0540/zzy/gpt-j/bm-gpt.pt")

    #init_student(model, teacher)
    #Trainer.forward = set_forward(teacher, Trainer.forward)

    bmp.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmp.synchronize()

    # data
    batch_size = 8
    dec_len = 512
    
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = bmp.optim.AdamOptimizer(model.parameters(), scale=2**20)
    lr_scheduler = bmp.lr_scheduler.Noam(optimizer, start_lr=1e-4, warmup_iter=2000, end_iter=100000)

    # for pruning
    # need to optimize the efficiency
    BMPrune.compute_mask(model, m4n2_2d_greedy, checkpoint="gpt-j-mask-int8.bin")
    BMPrune.set_optim_for_pruning(optimizer)

    # import BMMoE
    # num_expert, topk = 0
    # BMMoE.moefy(model, num_expert, topk)

    bmp.synchronize()
    average_time = 0
    average_time_shift = 0.9

    dataset = Dataset(
        MMapIndexedDataset("/data/home/scv0540/zzy/openwebtxt/openwebtext_text_document"),
        dec_len
    )

    # for eval
    model.eval()

    for _ in range(3):
        for iteration, data in enumerate(Trainer.batch_iter(dataset, batch_size, bmp.rank(), bmp.world_size())):

            if iteration < 20000:
                 continue

            if iteration >= 20100:
                 exit()

            # load data
            st = time.time()
            optimizer.zero_grad()

            # with bmp.inspect.inspect_tensor() as inspector:

            dec_input = data["ctx"].int()
            dec_length = data["len_ctx"].int()
            dec_mask = torch.arange(dec_len)[None, :].repeat(batch_size, 1) < dec_length[:, None]
            targets = torch.where(dec_mask, data["target"].long(), torch.scalar_tensor(-100, dtype=torch.long))

            targets = targets.cuda()
            dec_input = dec_input.cuda()
            dec_length = dec_length.cuda()

            # logits = model(dec_input, dec_length)
            # batch, seq_len, vocab_out_size = logits.size()

            # loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))
            loss, logits = Trainer.forward(model, dec_input, dec_length, targets, loss_func)
        
            global_loss = bmp.sum_loss(loss).item()
        
            loss = optimizer.loss_scale(loss)
            loss.backward()
        
            if iteration % 1000 == 0:
                print_inspect(model, "*")
            # bmp.print_rank(bmp.inspect.format_summary(inspector.get_summary()))
            
            #for eval
            #bmp.optim_step(optimizer, lr_scheduler)

            iteration_time = time.time() - st
            average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time
            bmp.print_rank(
                "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f}".format(
                    iteration,
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optimizer.scale),
                    average_time / (1 - pow(average_time_shift, iteration + 1))
                )
            )

            if iteration % 2000 == 0 and iteration != 0:
                bmp.save(model, "results-int8/ckpt-%d.pt" % iteration)
                # pass
    
    bmp.save(model, "checkpoint.pt")

if __name__ == '__main__':
    main()
