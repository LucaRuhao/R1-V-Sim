#!/bin/bash

# The latest vllm==0.7.2 is required for this script: pip3 install vllm==0.7.2 
echo "------------------ KILL GPU ----------------------"
pkill -9 pt_main_thread
pkill -9 python3
pkill -9 python

cd ../src/r1-v

export DEBUG_MODE="false"
export LOG_PATH="/cosmos/ruhao/code/R1-V-main/vllm_run.txt"

QWEN_PATH="/cosmos/ruhao/model/Qwen2-VL-2B-Instruct"
HF_DATASET="/scratch/ruhao/data/clevr_cogen_a_train"
OUTPUT_DIR="/scratch/ruhao/output"
RUN_NAME="Qwen2-VL-2B-GRPO-CLEVR-70k-vllm"

# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" torchrun --nproc_per_node="6" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py --use_vllm True \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $QWEN_PATH \
    --dataset_name $HF_DATASET \
    --deepspeed local_scripts/zero3_offload.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --temperature 1.0 \
    --num_generations 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels 400000 \
    --num_train_epochs 2 \
    --max_steps 13125 \
    --run_name $RUN_NAME \
    --save_steps 2000 \
    --save_only_model true

echo "------------------ START GPU ----------------------"
cd /cosmos/ruhao/code
cp -r /scratch/ruhao/output /cosmos/ruhao/model/r1-v
python3 DPO_train.py
