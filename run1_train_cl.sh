#!/bin/bash

export CUDA_VISIBLE_DEVICES=5   # 只用 1 张卡

# ==== WandB 离线配置 ====
export WANDB_MODE=offline                    # 离线模式，不联网、不弹登录
export WANDB_PROJECT=cl_memory_vanilla       # 项目名
export WANDB_API_KEY="dummy_offline_key"     # 给个非空字符串即可

# ==== WandB 日志根目录 ====
export WANDB_DIR="/workspace/CL_exp/wandblog/CL_4B_exp_data_id_78_512"  # ✅ 你想让 wandb 写到哪，就改这里
mkdir -p "$WANDB_DIR"                        # 确保目录存在

begin_id=0

for data_id in 7 8
do
    for ((ORDERR=$begin_id; ORDERR<15; ORDERR++))
    do
        echo "======== 开始训练 dataset_id=${data_id}, task_id=${ORDERR} ========"

        torchrun \
            --nproc_per_node=1 \
            --master_port=1238 \
            finetune_cl.py \
            --base_model '/workspace/Qwen3-4B' \
            --num_epochs 10 \
            --group_by_length \
            --lora_target_modules '["q_proj","v_proj"]' \
            --micro_batch_size 4 \
            --batch_size 32 \
            --max_input_length 512 \
            --dataset_id ${data_id} \
            --task_id ${ORDERR} \
            --output_path "/workspace/CL_exp/checkpoints/exp_3_4B_512/" \
            --wandb_project "${WANDB_PROJECT}"
    done
done
