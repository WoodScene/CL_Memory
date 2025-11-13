#!/bin/bash

# 设置起始变量 从0开始，14截至，不是15截至了
begin_id=0

for data_id in 1 2 3 4 5 6
do
    # 循环从 begin_id 到 15
    for ((ORDER=$begin_id; ORDER<14; ORDER++))
    do
        # 执行 Python 文件，传递参数 $i
        CUDA_VISIBLE_DEVICES=6 python generate_bwt.py \
            --base_model '/data_8T2/yujie/Backbones/Llama-2-7b-chat' \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \
            --method_name='vanilla' \
            --model_type='' \
            
        # 可以在这里添加任何你需要的其他操作，如等待一段时间等
    done
done