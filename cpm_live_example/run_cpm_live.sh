#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_PER_NODE=4

NNODES=1
MASTER_ADDR="localhost"
MASTER_PORT=12345
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $2 \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OPTS=""
OPTS+=" --model-config config/cpm_live_1B_pruned.json"
OPTS+=" --vocab-file vocab/vocab.txt"
OPTS+=" --batch-size 32"
OPTS+=" --train-iters 200000"
OPTS+=" --save-iters 500"
OPTS+=" --save-name cpm_live_checkpoint_300M_mask"
OPTS+=" --max-length 512"
OPTS+=" --save results/sprune/300M_mask"
OPTS+=" --lr 0.1"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 4.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step 74500"
OPTS+=" --log-dir logs/tensorboard/cpm_live_300M_mask"
OPTS+=" --load results/sprune/1B/cpm_live_checkpoint_1B-74500.pt"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} pretrain_cpm_live.py ${OPTS}"

echo ${CMD}
$CMD

