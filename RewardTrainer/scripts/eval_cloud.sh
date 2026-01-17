export CUDA_VISIBLE_DEVICES=5
python train/cloud/eval/eval.py \
    --model-path output/Qwen2.5-7B-Instruct-CLoud-HHRLHF \
    --benchmark data/rewardbench/filtered.json \
    --batch-size 8