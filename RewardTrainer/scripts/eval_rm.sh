#!/bin/bash
#SBATCH --job-name=yolov5  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --ntasks=4  # 需要多少个 cpu
#SBATCH --gres=gpu:1  #使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus  # 分区默认即可
export CUDA_VISIBLE_DEVICES="0" 
python -m eval.rewardbench.rewardbench \
    --model="/root/autodl-tmp/DistilRM/output/grm/Qwen2.5-3B-Instruct_diff_reward_len1024_lr1e-05/logs/checkpoint-375" \
    --dataset="data/eval/rewardbench.jsonl" \
    --output_dir="result/rewardbench" \
    --batch_size=32 \
    --max_length=1024 \
    --torch_dtype="bfloat16" \
    --load_json \
    --save_all \
