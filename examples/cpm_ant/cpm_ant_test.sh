export CUDA_VISIBLE_DEVICES=2,3
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:23333 cpm_ant/cpm_ant_test.py \
    --model-config /yinxr/gongbt/modelbase/cpm-ant/1B/cpm_live_1B_pruned.json \
    --load /yinxr/gongbt/modelbase/cpm-ant/1B/cpm_live_checkpoint_1B_pruned.pt \
    --teacher-config /yinxr/gongbt/modelbase/cpm-ant/10B/cpm_live_10B.json \
    --load-teacher /yinxr/gongbt/modelbase/cpm-ant/10B/cpm_live_checkpoint_10B.pt \
    --data-path /yinxr/cpmlive_data_bin/merged/ \
    --cook-config /yinxr/gongbt/BMCook/sprune_dev/examples/t5/configs/bmcook.json \