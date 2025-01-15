#!/bin/bash
echo "Contents of the current directory:"
ls -lah

export CUDA_VISIBLE_DEVICES=0
export GEO_BENCH_DIR=/mnt/data/cc_benchmark

model="dinov2_seg"
dataset="caffe"
task="seg"
batch_size="16"
lr="0.002"
epochs="30"
warmup_epochs="3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

/home/toolkit/.conda/envs/dofaEnv/bin/python src/main.py \
--output_dir logs/"${model}_${dataset}" \
--model ${model} \
--dataset ${dataset} \
--task ${task} \
--num_gpus ${num_gpus} \
--num_workers 8 \
--batch_size ${batch_size} \
--epochs ${epochs} \
--lr ${lr} \
--warmup_epochs ${warmup_epochs} \
--seed 42
