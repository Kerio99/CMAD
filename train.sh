#!/bin/bash

run_train() {
    local dataset="$1"
    local modality="$2"
    local output_dir="$3"
    shift 3
    local items=("$@")

    export CUDA_VISIBLE_DEVICES=8

    local normal_suffix="_normal.json"
    local outlier_suffix="_outlier.json"

    local normal_paths=()
    local outlier_paths=()

    local base_path="./data_processed_json/${dataset}"

    for item in "${items[@]}"; do
        normal_paths+=("$base_path/${item}${normal_suffix}")
        outlier_paths+=("$base_path/${item}${outlier_suffix}")
    done

    mkdir -p "${output_dir}" "${output_dir}/checkpoints"

    local cmd="python -W ignore main.py "
    cmd+="--normal_json_path"
    for normal in "${normal_paths[@]}"; do
        cmd+=" '$normal' "
    done

    cmd+="--outlier_json_path"
    for outlier in "${outlier_paths[@]}"; do
        cmd+=" '$outlier' "
    done

    cmd+="--modality ${modality} "
    cmd+="--layer_list '5' '15' '25' "
    cmd+="--output_dir '${output_dir}'"

    echo "[RUN] ${dataset} -> ${output_dir} (GPU 9)"
    eval $cmd
}

run_train mvtec rgb ./logs/train/trained_on_mvtec \
    bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper

run_train visa rgb ./logs/train/trained_on_visa \
    candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum

run_train btad rgb ./logs/train/trained_on_btad \
    01 02 03

run_train mpdd rgb ./logs/train/trained_on_mpdd \
    bracket_black bracket_brown bracket_white connector metal_plate tubes

run_train mvtec3d_depth depth ./logs/train/trained_on_mvtec3d_depth_part1 \
    bagel cable_gland carrot cookie dowel

run_train mvtec3d_depth depth ./logs/train/trained_on_mvtec3d_depth_part2 \
    foam peach potato rope tire

run_train eyecandies_normals rgb ./logs/train/trained_on_eyecandies_normals_part1 \
    CandyCane ChocolateCookie ChocolatePraline Confetto GummyBear

run_train eyecandies_normals rgb ./logs/train/trained_on_eyecandies_normals_part2 \
    HazelnutTruffle LicoriceSandwich Lollipop Marshmallow PeppermintCandy
