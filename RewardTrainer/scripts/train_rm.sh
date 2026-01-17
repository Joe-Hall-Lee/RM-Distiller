#!/bin/bash

export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1
dataset_name='/root/autodl-tmp/DistilRM/RewardTrainer/data/train/skywork_10k_qwen.json'
base_model='/root/autodl-tmp/DistilRM/models/Qwen2.5-3B-Instruct'
loss_type='margin'
log_dir='../output'
main_process_port=12541

n_gpu=2
learning_rate=1e-5
max_length=1024
num_train_epochs=1
per_device_train_batch_size=4
gradient_accumulation_steps=2

accelerate launch --num_processes ${n_gpu} \
    --main_process_port ${main_process_port} \
    --config_file ../accelerate/fsdp_config.yaml \
    train/reward_models/run_reward_models_train.py \
    --loss_type ${loss_type} \
    --base_model ${base_model} \
    --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_length ${max_length} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} \
    --bf16 True \
    --gradient_checkpointing True \
