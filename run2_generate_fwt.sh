#!/bin/bash

# 起始任务 id
begin_id=0

# 总任务数量=14（0~13）
end_id=15

# 自定义输出路径（你可以改成自己的）
output_root="/workspace/CL_exp/generate_ouputs/exp_3_4B"

for data_id in 7 8
do
    echo "====== Dataset ${data_id} 开始生成 FWT 推理结果 ======"

    for ((ORDER=$begin_id; ORDER<end_id; ORDER++))
    do
        echo ">>> 当前 Task = $ORDER"

        CUDA_VISIBLE_DEVICES=5 python generate_fwt.py \
            --base_model '/workspace/Qwen3-4B' \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \
            --method_name='fwt' \
            --prompt_template 'alpaca' \
            --lora_checkpoint_path "/workspace/CL_exp/checkpoints/exp_3_4B/" \
            --output_root "${output_root}"

        echo "--- Task $ORDER 生成完毕 ---"
        echo ""
    done
done
