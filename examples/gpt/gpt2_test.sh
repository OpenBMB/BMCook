export CUDA_VISIBLE_DEVICES=2,3
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost gpt/gpt2_test.py \
    --model gpt2-large \
    --save-dir /yinxr/gongbt/BMCook/sprune_dev/examples/results \
    --data-path /yinxr/gongbt/database/openwebtxt/openwebtext_text_document \
    --cook-config /yinxr/gongbt/BMCook/sprune_dev/examples/gpt/configs/gpt2-combine.json \