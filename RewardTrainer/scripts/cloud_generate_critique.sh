export CUDA_VISIBLE_DEVICES="5" 

python train/cloud/data/generate_self_critiques.py --model output/Qwen2.5-7B-Instruct-Critic-HHRLHF --base-dataset data/hhrlhf-skywork-cloud-dpo.json --output data/hhrlhf-skywork_critique_qwen.json