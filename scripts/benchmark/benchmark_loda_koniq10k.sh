#!/bin/bash
# loda on koniq10k
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/trainer.py \
        job=train_loda_koniq10k \
        split_index="${split_idx}"
done >> logs/benchmark_loda_koniq10k.log 2>&1
