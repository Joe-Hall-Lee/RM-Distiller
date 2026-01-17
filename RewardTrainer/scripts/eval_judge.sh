export CUDA_VISIBLE_DEVICES=0
python eval/run_bench.py \
    --name gpt-4o \
    --config configs/eval/gpt-4o.yaml \
    --benchmark rm-bench \
    --data-path data/eval
