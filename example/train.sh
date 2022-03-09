#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
module load cuda/10.2

model="gpt-relu"

results_dir="/data/home/scv0540/cyf/sparsity/sparsity-and-cook/results"
use_kd=1
kd_loss_scale=0
kd_use_mse=0
start_lr=0.01
init_std=0.02
init_with_teacher=0

# Save dir
save_dir="$results_dir/$model"
save_dir+="-lr${start_lr}"
save_dir+="-std${init_std}"

if [ $use_kd -eq 1 ]; then
  save_dir+="-kd"
  if [ $kd_use_mse -eq 1 ]; then
    save_dir+="-mse"
  fi
  save_dir+="-scale${kd_loss_scale}"
#else
#  save_dir+="-std${init_std}"
fi

# Custom suffix for output dir
if [ $1 != "" ]; then
  save_dir+="-$1"
fi

# Python command
cmd=" \
  train.py \
  --save-dir $save_dir \
  --start-lr $start_lr \
  --model $model \
  --init-std $init_std \
"


# Knowledge distillation
if [ $use_kd -eq 1 ]; then
  cmd+=" --use-kd "
  cmd+=" --kd-loss-scale ${kd_loss_scale} "
  cmd+=" --kd-temp 1.0 "
  if [ $kd_use_mse -eq 1 ]; then
    cmd+=" --kd-use-mse "
  fi
  if [ $init_with_teacher -eq 1 ]; then 
    cmd+=" --init-with-teacher "
  fi
fi
# end knowledge distillation


# Execute
mkdir -p $save_dir

logfile="$save_dir/log.txt"

final_cmd="torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost $cmd | tee $logfile"

echo $final_cmd
eval $final_cmd

