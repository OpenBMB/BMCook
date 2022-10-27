export CUDA_VISIBLE_DEVICES=2,3
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost t5/t5_test.py \
    --model t5-large \
    --save-dir ... \
    --data-path ... \
    --cook-config t5/configs/t5-combine.json \