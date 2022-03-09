#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
module load cuda/10.2

model="gpt-j"

results_dir="$(pwd)/results"
use_kd=1
kd_ce_logits=0
kd_mse_lasth=1
kd_mse_hidn=0
kd_mse_att=1
kd_mse_emb=0
kd_loss_scale=1
start_lr=0.0005  # for KD

# Save dir
save_dir="$results_dir/$model"
# save_dir+="-lr${start_lr}"

if [ $use_kd -eq 1 ]; then
  save_dir+="-kd"
  if [ $kd_ce_logits -eq 1 ]; then
    save_dir+="-logits"
  fi
  if [ $kd_mse_lasth -eq 1 ]; then
    save_dir+="-lasth"
  fi
  if [ $kd_mse_hidn -eq 1 ]; then
    save_dir+="-hidn"
  fi
  if [ $kd_mse_att -eq 1 ]; then
    save_dir+="-att"
  fi
  if [ $kd_mse_emb -eq 1 ]; then
    save_dir+="-emb"
  fi
  save_dir+="-scale${kd_loss_scale}"
fi

# Custom suffix for output dir
if [ "$1" != "" ]; then
  save_dir+="-$1"
fi

# Python command
cmd=" \
  train-kd.py \
  --save-dir $save_dir \
  --start-lr $start_lr \
  --model $model \
"


# Knowledge distillation
if [ $use_kd -eq 1 ]; then
  cmd+=" --use-kd "
  if [ $kd_ce_logits -eq 1 ]; then
    cmd+=" --kd-ce-logits "
  fi
  if [ $kd_mse_lasth -eq 1 ]; then
    cmd+=" --kd-mse-last-hidden "
  fi
  if [ $kd_mse_hidn -eq 1 ]; then
    cmd+=" --kd-mse-hidn "
  fi
  if [ $kd_mse_att -eq 1 ]; then
    cmd+=" --kd-mse-att "
  fi
  if [ $kd_mse_emb -eq 1 ]; then
    cmd+=" --kd-mse-emb "
  fi
  cmd+=" --kd-loss-scale $kd_loss_scale "
fi
# end knowledge distillation


# Execute
mkdir -p $save_dir

logfile="$save_dir/log.txt"

final_cmd="torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost $cmd | tee $logfile"

echo $final_cmd
eval $final_cmd

