#!/bin/bash
# loda on live
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/trainer.py \
        job=train_loda_live \
        split_index="${split_idx}"
done >> logs/benchmark_loda_live.log 2>&1
