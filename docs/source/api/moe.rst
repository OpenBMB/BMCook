====================
MoEfication
====================

BMMoE
==========================================


To use this module, you need to implement router operation in FFNs as follows:

.. code-block:: python

   if self.moe is not None:
   with torch.no_grad():
         xx_ = input.float().transpose(1,2).reshape(-1, hidden_size)
         xx = xx_ / torch.norm(xx_, dim=-1).unsqueeze(-1)

         score = self.markers(xx)
         labels = torch.topk(score, k=self.k, dim=-1)[1].reshape(bsz, seq_len, self.k)
         cur_mask = torch.nn.functional.embedding(labels, self.patterns).sum(-2).transpose(1,2).detach()

.. code-block:: python
   
   if self.moe is not None:
      inter_hidden[cur_mask == False] = 0

.. autoclass:: moe.BMMoE
   :members:
