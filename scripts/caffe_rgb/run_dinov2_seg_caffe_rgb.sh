#!/bin/bash
echo "Contents of the current directory:"
ls -lah

export GEO_BENCH_DIR=/mnt/data/cc_benchmark

model="dinov2_seg"
dataset="caffe_rgb"
task="segmentation"
batch_size="16"
lr_min="0.0001"
lr_max="0.003"
epochs="30"
warmup_epochs="3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

/home/toolkit/.conda/envs/dofaEnv/bin/python src/hparam_ray.py \
--output_dir /mnt/results/nils/exps/hparams/${model}_${dataset} \
--model ${model} \
--dataset ${dataset} \
--task ${task} \
--num_workers 4 \
--epochs ${epochs} \
--seed 42 \
--num_gpus 1 \
--warmup_epochs ${warmup_epochs} \
--lr_min ${lr_min} \
--lr_max ${lr_max} \
--batch_choices 16 32