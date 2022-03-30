import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--save-dir', required=True, help='Directory to save result to')
    p.add_argument('--log-interval', type=int, default=1, help='Log loss, LR, scale every n steps')
    p.add_argument('--save-interval', type=int, default=1000, help='Interval steps between checkpoints')

    # distillation
    p.add_argument('--use-kd', action='store_true', help='Use knowledge distillation')
    p.add_argument('--kd-loss-scale', type=float, default=1.0, help='KD loss is multiplied by this')
    p.add_argument('--kd-temp', type=float, default=1.0, help='Temperature in KD')
    p.add_argument('--kd-ce-logits', action='store_true', help='Use CE on teacher logits as KD loss')
    p.add_argument('--kd-mse-last-hidden', action='store_true', help='Use MSE on the last hidden states as KD loss')
    p.add_argument('--kd-mse-hidn', action='store_true', help='Use MSE on the all hidden states as KD loss. See TinyBERT')
    p.add_argument('--kd-mse-att', action='store_true', help='Use MSE on the all attention scores as KD loss. See TinyBERT')
    p.add_argument('--kd-mse-emb', action='store_true', help='Use MSE on the embeddings as KD loss')
    p.add_argument('--init-with-teacher', action='store_true', help='Initialize student with teacher')
    p.add_argument('--load-teacher', type=str, default='/data/home/scv0540/zzy/gpt-j/bm-gpt.pt', help='Load teacher checkpoint')

    # pruning
    p.add_argument('--use-pruning', action='store_true', help='Use pruning')
    p.add_argument('--pruning-mask-path', type=str, default="gpt-j-mask.bin", help='Path to pruning mask')
    p.add_argument('--sprune', action='store_true', help='Structure pruning')
    p.add_argument('--original-model', type=str, help='Path to the checkpoint of the original model')

    # Hyperparameters
    p.add_argument('--start-lr', type=float, default=0.01, help='Start learning rate of inverse square root')
    p.add_argument('--init-std', type=float, default=0.02, help="Standard deviation of normal distribution for initializing model parameters")
    p.add_argument('--resuming-step', type=int, default=0, help="Continue training from a specific step")

    # evaluation
    p.add_argument('--eval', action='store_true', help='Turn off dropout and evaluate the model on the test set')
    p.add_argument('--load', type=str, default="", help='Load checkpoint')

    # MoE
    p.add_argument('--save-hidden', action='store_true', help='Save hidden states for MoEfication')
    p.add_argument('--moe', action='store_true', help='Simulate MoE')
    p.add_argument('--moe-ckpt', type=str, default="", help='Path to MoE router')
    p.add_argument('--num-expert', type=int, default=512, help='Number of experts')
    p.add_argument('--topk', type=int, default=102, help='Select top-k experts')

    return p.parse_args()

