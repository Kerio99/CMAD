#!/bin/bash

export CUDA_VISIBLE_DEVICES=9

# 声明一个关联数组（字典）
declare -A test_dataset_modality_dict

# 为每个数据集指定模态
test_dataset_modality_dict=(
                            ["mvtec"]="rgb"
                            ["visa"]="rgb"
                            ["btad"]="rgb"
                            ["mpdd"]="rgb"
                            ["mvtec3d_depth"]="depth"
                            ["eyecandies_normals"]="depth"
                            ["brainad"]="medical"
                            ["retina_oct2017"]="medical"
                            ["wtb"]="thermal"
                            ["leaves"]="thermal"
                        )

# 按顺序声明一个数组
test_dataset_name_list=(
    "mvtec"
    "visa"
    "btad"
    "mpdd"

    "mvtec3d_depth"
    "eyecandies_normals"

    "brainad"
    "retina_oct2017"

    "wtb"
    "leaves"    
)

shot_list=(
    "1"
    "2"
    "4"
    "8"
)

for shot in "${shot_list[@]}"; do
    for test_dataset_name in "${test_dataset_name_list[@]}"; do
        modality=${test_dataset_modality_dict[$test_dataset_name]}

        # 定义输出文件
        output_result_file="./logs/test_results/${shot}-shot"

        cmd="python -W ignore test.py"

        cmd+=" --test_dataset_name ${test_dataset_name}"
        cmd+=" --modality ${modality}"
        cmd+=" --shot ${shot}"
        cmd+=" --output_result_file ${output_result_file}"
        cmd+=" --prototype_list_path ./prototype_list.pt"
        cmd+=" --data_processed_json_root "
        cmd+=" --layer_list '5' '15' '25' "

        eval $cmd &
        wait
    done
done
