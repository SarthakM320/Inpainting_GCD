#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name 'officehome' \
    --batch_size 8 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --exp_name trial
