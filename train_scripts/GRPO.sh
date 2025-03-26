echo "------------------ KILL GPU ----------------------"
pkill -9 pt_main_thread
pkill -9 python3
pkill -9 python

cd ../src/r1-v

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="/cosmos/ruhao/code/R1-V-main/debug_log_2b.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12341" \
    src/open_r1/grpo.py \
    --output_dir "/scratch/ruhao/r1-v-grpo" \
    --model_name_or_path "/cosmos/ruhao/model/Qwen2-VL-2B-Instruct" \
    --dataset_name "/scratch/ruhao/data/clevr_cogen_a_train" \
    --deepspeed local_scripts/zero3_offload.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k-test \
    --save_steps 400 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance

echo "------------------ START GPU ----------------------"
cd /cosmos/ruhao/code
#cp -r /scratch/ruhao/r1-v-grpo/ /cosmos/ruhao/model/r1-v
python3 DPO_train.py