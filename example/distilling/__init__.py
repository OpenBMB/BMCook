import torch
from torch import nn
import bmtrain as bmp
import torch.nn.functional as F
import cpm_kernels.torch as ct


class HiddenMap(bmp.DistributedModule):
    def __init__(self, dim_from, dim_to, dtype=torch.half, 
                 init_std=0.02, int8=False):
        super().__init__()

        self.dtype = dtype
        self.int8 = int8

        init_method = bmp.ParameterInitializer(
            nn.init.normal_, mean=0.0, std=init_std)
        self.map = bmp.DistributedParameter(
            torch.empty(dim_to, dim_from, dtype=dtype), 
            init_method=init_method)
    
    def forward(self, x):
        map = self.map
        # Map hidden states from student to teacher
        # (1#batch, dim_to, dim_from) @ (batch, dim_from, seq_len) = (batch, dim_to, seq_len)
        x = ct.bmm(map.unsqueeze(0), False, x, False, int8=self.int8)
        return x


class BMDistill:
    '''
    BMDistill provide additional training objectives for knowledge distillation, which further improves the performance of compressed models.
    '''

    @classmethod
    def init_student(cls, student, teacher_sd):
        '''

        Initialize the student model based on the parameters of the teacher model, i.e, copy every 2 layer from teacher to student.

        :param student: Student model.
        :param teacher_sd: State dictionary of the teacher model.

        '''
        # teacher_sd = teacher.state_dict()
        student_sd = {}
        for k, weights in teacher_sd.items():
            if 'dec_layers' not in k:
                student_sd[k] = weights
            else:
                layer = int(k.split('.')[1])
                if layer % 2 == 0:
                    new_layer = str(layer // 2)
                    new_k = k.replace(str(layer), new_layer)
                    student_sd[new_k] = weights
        student.load_state_dict(student_sd)

    @classmethod
    def set_forward(cls, student, teacher, foward_fn,
                    temp=1.0, kd_loss_scale=1.0, 
                    output_kd_loss=False,
                    int8=False,
                    # Various KD loss
                    ce_logits=False,
                    mse_last_hidden=False,
                    mse_hidden_states=False,
                    mse_att=False,
                    mse_emb=False):
        '''
        Modify the forward function of the student model to compute additional knowledge distillation loss.

        `student` and `teacher` should return (logits, hidden_states, att_scores).
        logits: (batch_size, vocab_size)
        hidden_states: (batch_size, dec_len, hidden_size)
        att_scores: (batch_size, dec_len, enc_len)

        :param student: Student model.
        :param teacher: Teacher model.
        :param foward_fn: Forward function of the student model.
        :param temp: Temperature for knowledge distillation.
        :param kd_loss_scale: Scale of the knowledge distillation loss.
        :param output_kd_loss: Whether to output the knowledge distillation loss.
        :param ce_logits: Whether to use cross entropy loss for the output logits.
        :param mse_last_hidden: Whether to use MSE loss for the last hidden state.
        :param mse_hidden_states: Whether to use MSE loss for the hidden states.
        :param mse_att: Whether to use MSE loss for the attention matrices.
        :param mse_emb: Whether to use MSE loss for the embedding matrices.
        :return: Modified forward function, whose return values are 1. (model, dec_input, dec_length, targets) -> loss, logits, kd_loss 2. (model, dec_input, dec_length, targets) -> loss, logits. Returns 1 if `output_kd_loss` is True, else return 2.
        '''

        cls.int8 = int8

        assert any([ce_logits, mse_last_hidden, mse_hidden_states, mse_att, mse_emb])

        if mse_hidden_states:
            cls.hidden_map = HiddenMap(teacher.dim_model, student.dim_model)
            bmp.init_parameters(cls.hidden_map)
            bmp.synchronize()

        def forward(model, dec_input, dec_length, targets, loss_func):
            outputs = foward_fn(
                model, dec_input, dec_length, targets, loss_func)
            loss = outputs[0]
            model_outputs = outputs[1:]
            logits_s = model_outputs[0]
            hidden_s = model_outputs[1]
            att_scores_s = model_outputs[2]

            outputs_t = teacher(dec_input, dec_length)
            logits_t = outputs_t[0].detach()
            hidden_t = outputs_t[1]
            att_scores_t = outputs_t[2]

            # Compute loss and d_loss
            d_loss = 0.0
            if ce_logits:
                prob_t = F.softmax(logits_t / temp, dim=-1)
                log_prob_s = F.log_softmax(logits_s / temp, dim=-1)
                d_loss += -(prob_t * log_prob_s).sum(dim=1).mean()

            # MSE loss on last hidden states
            if mse_last_hidden:
                h_t = hidden_t[-1].detach()
                h_s = hidden_s[-1]
                d_loss += F.mse_loss(h_s, h_t)

            # MSE loss on all hidden states
            if mse_hidden_states:
                cls.hidden_map.to(dec_input.device)
                ratio = len(hidden_t) // len(hidden_s)
                fit_target = [hidden_t[i]
                    for i in range(ratio - 1, len(hidden_t), ratio)]
                for h_s, h_t in zip(hidden_s, fit_target):
                    h_t = h_t.detach()
                    # Map hidden states from student to teacher
                    h_s = cls.hidden_map(h_s)
                    d_loss += F.mse_loss(h_s, h_t)

            # MSE loss on attention scores
            if mse_att:
                ratio = len(hidden_t) // len(hidden_s)
                fit_target = [att_scores_t[i]
                    for i in range(ratio - 1, len(att_scores_t), ratio)]
                for att_s, att_t in zip(att_scores_s, fit_target):
                    att_t = att_t.detach()
                    att_t = att_t[att_t != -torch.inf]
                    att_s = att_s[att_s != -torch.inf]
                    d_loss += F.mse_loss(att_s, att_t)

            loss = loss + kd_loss_scale * d_loss

            outputs = (loss, logits_s)
            if output_kd_loss:
                outputs = outputs + (d_loss, )
            return outputs
        return forward
