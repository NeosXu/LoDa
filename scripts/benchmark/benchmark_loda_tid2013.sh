#!/bin/bash
# loda on tid2013
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/trainer.py \
        job=train_loda_tid2013 \
        split_index="${split_idx}"
done >> logs/benchmark_loda_tid2013.log 2>&1
