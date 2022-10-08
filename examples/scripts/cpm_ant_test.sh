export CUDA_VISIBLE_DEVICES=1,2
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:23333 cpm_ant_test.py 