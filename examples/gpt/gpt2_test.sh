export CUDA_VISIBLE_DEVICES=2,3
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost gpt/gpt2_test.py \
    --model gpt2-large \
    --save-dir ... \
    --data-path ... \
    --cook-config gpt/configs/gpt2-prune.json \