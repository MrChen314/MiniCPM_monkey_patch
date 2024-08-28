#!/bin/bash

# 获取当前目录的名字
current_dir=$(basename "$PWD")

# 判断当前目录名字是否为 "finetune"
if [ "$current_dir" = "finetune" ]; then
  # 如果在finetune目录下，则返回上一级目录
  cd ..
fi

source /home/c30061641/cann-b030/ascend-toolkit/set_env.sh

USE_FLASH_ATTENTION_2=true
export use_flash_attention_2=$USE_FLASH_ATTENTION_2

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/home/c30061641/MiniCPM-V/data/model" # or openbmb/MiniCPM-V-2
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
#DATA="/home/c30061641/MiniCPM-V/data/TextVQA/TextVQA_0.5.1_train_format.json"
DATA="/home/c30061641/MiniCPM-V/data/TextVQA/TextVQA_0.5.1_train_balance.json"
EVAL_DATA="/home/c30061641/MiniCPM-V/data/TextVQA/TextVQA_0.5.1_val_format.json"
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune/finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm true \
    --dataloader_num_workers 1 \
    --model_max_length 2048 \
    --max_slice_nums 9 \
    --max_steps 10000 \
    --eval_steps 20000 \
    --output_dir finetune/output/output_minicpmv2 \
    --logging_dir finetune/output/output_minicpmv2 \
    --logging_strategy "steps" \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 10 \
    --learning_rate 5e-7 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed finetune/ds_config_zero2.json \
    --report_to "tensorboard" 2>&1 | tee finetune/log/fpft_performance/npu.log 2>&1 &
