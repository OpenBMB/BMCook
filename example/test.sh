#!/bin/bash
#SBATCH -G 2
#SBATCH -p rtx2080
num_gpus=2
grad_acc=8
# data_path="/home/chenyingfa/sparsity-and-cook/tools/openwebtext_text_document"
data_path="/home/chenyingfa/sparsity-and-cook/tools/wikitext-103"
A=8
L=6
H=512
# save_dir="/home/chenyingfa/sparsity-and-cook/results/gpt-relu-h${H}-l${L}-a${A}"
save_dir="/home/chenyingfa/sparsity-and-cook/results/gpt-relu-h${H}-l${L}-a${A}-wikitext"
mode="test"
huggingface=1

batch_size=$((128 / ${num_gpus} / ${grad_acc}))

mkdir -p ${save_dir}

cmd="\
	train.py \
	--save-dir ${save_dir} \
	--dynamic-save-interval \
	--hidden-dim ${H} \
	--layers ${L} \
	--attn-heads ${A} \
	--data-path ${data_path} \
	--batch-size ${batch_size} \
	--grad-acc ${grad_acc} \
"

if [[ $huggingface -eq 1 ]]; then
	cmd+=" --huggingface"  # TODO: remove
	save_dir+="-huggingface"
fi

if [[ $mode == *"train"* ]]; then
	cmd="${cmd} --train"
fi

if [[ $mode == *"test"* ]]; then
	cmd="${cmd} --test"
fi

logfile="$save_dir/testlog.txt"

mkdir -p $save_dir

echo "Running command: ${cmd}"
torchrun --nnodes=1 --nproc_per_node=${num_gpus} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost ${cmd} | tee ${logfile}
