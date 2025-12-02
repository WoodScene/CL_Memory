#!/usr/bin/env bash
set -e

output_root="/workspace/CL_exp/generate_ouputs/exp_3_4B"

# 这里 dataset_id=7，是你现在的实验
for data_id in 7 8
do
    CUDA_VISIBLE_DEVICES=4 python generate_avgPerf.py \
        --base_model "/workspace/Qwen3-4B" \
        --dataset_id=${data_id} \
        --method_name "cl" \
        --model_type "" \
        --prompt_template "alpaca" \
        --max_task_id 14 \
        --use_latest_checkpoint False \
        --lora_checkpoint_path "/workspace/CL_exp/checkpoints/exp_3_4B/" \
        --output_root "${output_root}"
done
