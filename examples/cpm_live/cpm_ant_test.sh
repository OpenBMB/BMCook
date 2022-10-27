export CUDA_VISIBLE_DEVICES=2,3
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:23333 cpm_live/cpm_ant_test.py \
    --model-config ... \
    --load ... \
    --teacher-config ... \
    --load-teacher ... \
    --data-path ... \
    --cook-config cpm_live/configs/prune.json \