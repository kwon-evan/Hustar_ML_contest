#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='eval'
ROOT=/home/piai/다운로드/U2PL-main

mkdir -p log
mkdir -p checkpoints/results

python $ROOT/infer.py \
    --config=config.yaml \
    --base_size 2048 \
    --scales 1.0 \
    --model_path=experiments/cityscapes/744/ours/checkpoints/ckpt.pth \
    --save_folder=experiments/cityscapes/744/ours/checkpoints/results \
    2>&1 | tee log/val_best_$now.txt
