#!/bin/bash
echo "Contents of the current directory:"
ls -lah

export CUDA_VISIBLE_DEVICES=0
export GEO_BENCH_DIR=/mnt/data/cc_benchmark

model="anysat_cls"
dataset="benv2_rgb"
task="classification"
batch_size="16"
lr_min="0.001"
lr_max="0.1"
epochs="30"
warmup_epochs="3"
batch_choices="64 128"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

/home/toolkit/.conda/envs/dofaEnv/bin/python src/hparam_ray.py \
--output_dir /mnt/results/nils/exps/${model}_${dataset} \
--model ${model} \
--dataset ${dataset} \
--task ${task} \
--num_gpus ${num_gpus} \
--num_workers 8 \
--epochs ${epochs} \
--warmup_epochs ${warmup_epochs} \
--seed 42 \
--lr_min ${lr_min} \
--lr_max ${lr_max} \
--batch_choices ${batch_choices} \
