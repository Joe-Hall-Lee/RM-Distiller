export CUDA_VISIBLE_DEVICES=0,2,4,6

composer -n 4 train/cloud/train/train.py train/cloud/train/configs/8b_cloud.yaml