#!/bin/bash

# 起始任务 id
begin_id=0

# 总任务数量=14（0~13）
end_id=15

# 自定义输出路径（你可以改成自己的）
output_root="/workspace/CL_exp/generate_ouputs/exp_1_0d6B"

for data_id in 7 8
do
    echo "====== Dataset ${data_id} 开始生成 BWT 推理结果 ======"

    for ((ORDER=$begin_id; ORDER<end_id; ORDER++))
    do
        echo ">>> 当前 Task = $ORDER"

        CUDA_VISIBLE_DEVICES=5 python generate_bwt.py \
            --base_model "/workspace/Qwen3-0.6B" \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \
            --method_name='cl' \
            --model_type='' \
            --prompt_template 'alpaca' \
            --lora_checkpoint_path "/workspace/CL_exp/checkpoints/exp_1_0d6B_512/" \
            --output_root "${output_root}"

        echo "--- Task $ORDER 生成完毕 ---"
        echo ""
    done
done
