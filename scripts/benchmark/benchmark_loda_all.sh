#!/bin/bash
# loda on koniq10k
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/trainer.py \
        job=train_loda_koniq10k \
        split_index="${split_idx}"
done >> logs/benchmark_loda_koniq10k.log 2>&1

# loda on kadid10k
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/trainer.py \
        job=train_loda_kadid10k \
        split_index="${split_idx}"
done >> logs/benchmark_loda_kadid10k.log 2>&1

# loda on spaq
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/trainer.py \
        job=train_loda_spaq \
        split_index="${split_idx}"
done >> logs/benchmark_loda_spaq.log 2>&1

# loda on livec
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/trainer.py \
        job=train_loda_livec \
        split_index="${split_idx}"
done >> logs/benchmark_loda_livec.log 2>&1

# loda on live
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/trainer.py \
        job=train_loda_live \
        split_index="${split_idx}"
done >> logs/benchmark_loda_live.log 2>&1

# loda on tid2013
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/trainer.py \
        job=train_loda_tid2013 \
        split_index="${split_idx}"
done >> logs/benchmark_loda_tid2013.log 2>&1

# loda on flive
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/trainer.py \
        job=train_loda_flive \
        split_index="${split_idx}"
done >> logs/benchmark_loda_flive.log 2>&1
