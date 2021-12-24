#!/bin/bash

set +x

OUTPUT_DIR="klue_output"
DATA_DIR="data/klue_benchmark"  # default submodule for data from https://github.com/KLUE-benchmark/KLUE
VERSION="v1.1"

# KLUE-DP
task="klue-dp"
for model_name in "klue/roberta-base"; do
    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path ${model_name} --learning_rate 5e-5 --num_train_epochs 10 --warmup_ratio 0.1 --train_batch_size 32 --patience 100000 --max_seq_length 256 --metric_key las_macro_f1 --gpus 3 --num_workers 4
done

# python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-large --learning_rate 5e-5 --num_train_epochs 15 --gradient_accumulation_steps 1 --warmup_ratio 0.2 --train_batch_size 32 --patience 10000 --max_seq_length 256 --metric_key uas_macro_f1 --gpus 3 --num_workers 4


# for multi gpu
#for model_name in "klue/roberta-small" "klue/roberta-base" "klue/bert-base"; do
#    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path ${model_name} --learning_rate 3e-5 --num_train_epochs 10 --train_batch_size 32 --eval_batch_size 32 --max_seq_length 510 --gradient_accumulation_steps 1 --warmup_ratio 0.2 --weight_decay 0.01 --max_grad_norm 1.0 --patience 100000 --metric_key slot_micro_f1 --gpus 0 1 --num_workers 8
#done

#python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-large --learning_rate 3e-5 --num_train_epochs 10 --train_batch_size 16 --eval_batch_size 16 --max_seq_length 510 --gradient_accumulation_steps 2 --warmup_ratio 0.2 --weight_decay 0.01 --max_grad_norm 1.0 --patience 100000 --metric_key slot_micro_f1 --gpus 0 1 --num_workers 8

