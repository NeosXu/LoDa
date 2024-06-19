#!/bin/bash
# loda on koniq10k
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
        job=eval \
        run_group=loda_koniq10k_eval \
        name=loda_koniq10k_eval_split"${split_index}" \
        split_index="${split_idx}" \
        data=koniq10k \
        load.network_chkpt_path=chkpt/rep/koniq10k/loda_koniq10k_split"${split_idx}".pt
done >> logs/loda_koniq10k_eval.log 2>&1

# loda on kadid10k
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
        job=eval \
        run_group=loda_kadid10k_eval \
        name=loda_kadid10k_eval_split"${split_index}" \
        split_index="${split_idx}" \
        data=kadid10k \
        load.network_chkpt_path=chkpt/rep/kadid10k/loda_kadid10k_split"${split_idx}".pt
done >> logs/loda_kadid10k_eval.log 2>&1

# loda on livec
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
        job=eval \
        run_group=loda_livec_eval \
        name=loda_livec_eval_split"${split_index}" \
        split_index="${split_idx}" \
        data=livec \
        load.network_chkpt_path=chkpt/rep/livec/loda_livec_split"${split_idx}".pt
done >> logs/loda_livec_eval.log 2>&1

# loda on live
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
        job=eval \
        run_group=loda_live_eval \
        name=loda_live_eval_split"${split_index}" \
        split_index="${split_idx}" \
        data=live \
        load.network_chkpt_path=chkpt/rep/live/loda_live_split"${split_idx}".pt
done >> logs/loda_live_eval.log 2>&1

# loda on spaq
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
        job=eval \
        run_group=loda_spaq_eval \
        name=loda_spaq_eval_split"${split_index}" \
        split_index="${split_idx}" \
        data=spaq \
        load.network_chkpt_path=chkpt/rep/spaq/loda_spaq_split"${split_idx}".pt
done >> logs/loda_spaq_eval.log 2>&1

# loda on tid2013
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
        job=eval \
        run_group=loda_tid2013_eval \
        name=loda_tid2013_eval_split"${split_index}" \
        split_index="${split_idx}" \
        data=tid2013 \
        load.network_chkpt_path=chkpt/rep/tid2013/loda_tid2013_split"${split_idx}".pt
done >> logs/loda_tid2013_eval.log 2>&1

# loda on flive
for split_idx in {0..9};
do
    CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
        job=eval \
        run_group=loda_flive_eval \
        name=loda_flive_eval_split"${split_index}" \
        split_index="${split_idx}" \
        data=flive \
        load.network_chkpt_path=chkpt/rep/flive/loda_flive_split"${split_idx}".pt
done >> logs/loda_flive_eval.log 2>&1
