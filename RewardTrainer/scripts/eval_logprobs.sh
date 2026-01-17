export CUDA_VISIBLE_DEVICES="0" 
python -m eval.rewardbench.rewardbench \
    --model="../output/Qwen2.5-3B-Instruct-orpo_1" \
    --ref_free_norm="avg" \
    --dataset="data/eval/part_2_greedy.json" \
    --output_dir="result/part_2_greedy" \
    --batch_size=16 \
    --load_json \
    --save_all