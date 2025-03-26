# R1-V-Sim

原始项目：https://github.com/Deep-Agent/R1-V

run on CUDA_12.4 torch-2.6.0-cp311

creat env
```shell
conda create -n r1-v python=3.11
conda activate r1-v

bash setup.sh
```

download model / dataset
```shell
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct
huggingface-cli download --repo-type dataset --resume-download leonardPKU/clevr_cogen_a_train
```

change Qwen2VL code
```shell
rm modeling_qwen2_vl.py Code(such as /home/.conda/envs/r1-v/lib/python3.11/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py)
cp ./qwen2_vl/modeling_qwen2_vl.py  modeling_qwen2_vl.py Code Path
```

run GRPO
```shell
cd ../src/r1-v

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir <OUTPUT_DIR> \
    --model_name_or_path <PATH-TO-Qwen2-VL-2B-Instruct> \ 
    --dataset_name leonardPKU/clevr_cogen_a_train \  
    --deepspeed local_scripts/zero3_offload.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 500 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  

```


所有代码的修改都在： 


src/r1-v/src/open_r1/grpo.py

src/r1-v/src/open_r1/trainer/grpo_trainer.py