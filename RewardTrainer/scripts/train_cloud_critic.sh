export CUDA_VISIBLE_DEVICES=1,3,5,7

composer -n 4 train/cloud/train/train.py train/cloud/train/configs/8b_critique_sft.yaml