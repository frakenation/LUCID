#!/usr/bin/env bash
set -e

export PYTHONPATH=.
python -m src.inference_cfg_index \
    --dataset_config_path dataloader/dataset_test.yml \
    --pretrained_model_name_or_path stabilityai/sd-turbo \
    --model_path ./lucid_checkpoints/model_40000.pkl \
    --flare_disentanglement_path ./checkpoints/flare_disentanglement/latest.pth \
    --resolution 512 \
    --batch_size 1 \
    --num_samples 1500 \
    --num_workers 0 \
    --positive_prompt "light source, enhance this low-light image with better illumination and clarity" \
    --negative_prompt "light source, add noise and artifacts" \
    --cfg_scales "0.25,0.50,0.75,0.95,1.05,1.25,1.50" \
    --output_dir ./results/lucid_cfg_index \
    --timestep 199 \
    --ms_unet \
    --enable_hdr_fusion \
    --visualize_weights \
    --tonemap_method drago \
    --device cuda \
    --seed 42
