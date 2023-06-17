#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python main.py \
        --do_train \
        --output_dir out/gt2_pq \
        --train_file data/pq/train \
        --predict_file data/pq/val	\
        --model_path pretrain_model/jointgt_bart \
        --tokenizer_path pretrain_model/jointgt_bart \
        --dataset pq \
        --train_batch_size 32 \
        --predict_batch_size 32 \
        --max_input_length 256 \
        --max_output_length 128 \
        --append_another_bos \
        --learning_rate 2e-5 \
        --num_train_epochs 200 \
        --warmup_steps 300 \
        --eval_period 100 \
        --num_beams 5 \
        --wait_step 50 \
        --seed 42 \
