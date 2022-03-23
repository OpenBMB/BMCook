#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
module load cuda/10.2

logfile="log_distill_prune.txt"

# torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py --save-dir /data/home/scv0540/zzy/gpt-j/example/results/gpt-j-relu --model gpt-j-full-relu --use-kd --kd-mse-last-hidden --start-lr 1e-4 --kd-loss-scale 1 | tee $logfile

#torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py --save-dir /data/home/scv0540/zzy/gpt-j/example/results/gpt-j-int8-prune-new --model gpt-j-full-int8 --use-kd --kd-mse-last-hidden --start-lr 1e-4 --kd-loss-scale 1 --load /data/home/scv0540/zzy/gpt-j/bm-gpt.pt --use-pruning --pruning-mask-path /data/home/scv0540/zzy/bmcook-dev/example/gpt-j-mask.bin | tee $logfile
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py --save-dir /data/home/scv0540/zzy/gpt-j/example/results/gpt-j-prune-new --model gpt-j-full --use-kd --kd-mse-last-hidden --start-lr 1e-4 --kd-loss-scale 1 --load /data/home/scv0540/zzy/gpt-j/bm-gpt-new.pt --load-teacher /data/home/scv0540/zzy/gpt-j/bm-gpt-new.pt --use-pruning --pruning-mask-path /data/home/scv0540/zzy/bmcook-dev/example/gpt-j-mask.bin | tee $logfile

# new
# torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py --save-dir /data/home/scv0540/zzy/gpt-j/example/results/test --model gpt-j-full-int8 --load /data/home/scv0540/zzy/gpt-j/example/results/gpt-j-int8-prune-new/checkpoints/ckpt-8000.pt --use-pruning --pruning-mask-path /data/home/scv0540/zzy/bmcook-dev/example/gpt-j-mask.bin --eval | tee $logfile

# old
#torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py --save-dir /data/home/scv0540/zzy/gpt-j/example/results/test --model gpt-j-full-int8 --load /data/home/scv0540/zzy/gpt-j/example/results/gpt-j-int8-prune-distill/checkpoints/ckpt-7000.pt --use-pruning --pruning-mask-path /data/home/scv0540/zzy/bmcook-dev/example/gpt-j-mask.bin --eval | tee $logfile

#torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py --save-dir /data/home/scv0540/zzy/gpt-j/example/results/test --model gpt-j-full --load /data/home/scv0540/zzy/gpt-j/bm-gpt-new.pt --eval

# torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py --save-dir /data/home/scv0540/zzy/gpt-j/example/results/gpt-j-sprune-int8-prune-distill --model gpt-j-int8 --start-lr 1e-4 --use-pruning --pruning-mask-path gpt-j-sp-mask.bin --use-kd --kd-mse-last-hidden --kd-loss-scale 1 --init-with-teacher | tee $logfile

 #--eval --load /data/home/scv0540/zzy/gpt-j/example/results/gpt-j-prune-kd/checkpoints/ckpt-6000.pt

#torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost train.py --save-dir /data/home/scv0540/zzy/gpt-j/example/results/test-1 --model gpt-j-full-int8 --start-lr 1e-4 --use-pruning --eval --load /data/home/scv0540/zzy/gpt-j/example/results/gpt-j-int8-prune-distill/checkpoints/ckpt-6000.pt
