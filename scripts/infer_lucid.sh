#!/usr/bin/env bash
set -e

export PYTHONPATH=.
python -m src.inference \
    --dataset_config_path dataloader/dataset_test.yml \
    --pretrained_model_name_or_path stabilityai/sd-turbo \
    --model_path ./lucid_checkpoints/model_40000.pkl \
    --flare_disentanglement_path ./checkpoints/flare_disentanglement/latest.pth \
    --resolution 512 \
    --batch_size 1 \
    --num_samples 1500 \
    --num_workers 0 \
    --inference_mode cfg_guidance \
    --positive_prompt "light source, enhance this low-light image with better illumination and clarity" \
    --negative_prompt "light source, add noise and artifacts" \
    --output_dir ./results/lucid_cfg_150 \
    --timestep 199 \
    --ms_unet \
    --cfg_scale 1.5 \
    --device cuda \
    --seed 42
