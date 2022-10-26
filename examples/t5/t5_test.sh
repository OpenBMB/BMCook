export CUDA_VISIBLE_DEVICES=2,3
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost t5/t5_test.py \
    --model t5-large \
    --save-dir /yinxr/gongbt/BMCook/sprune_dev/examples/results \
    --data-path /yinxr/gongbt/database/t5/pretrain_pile_data \
    --cook-config /yinxr/gongbt/BMCook/sprune_dev/examples/t5/configs/bmcook.json \