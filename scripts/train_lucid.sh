#!/usr/bin/env bash
set -e

export PYTHONPATH=.
python -m src.train_lucid \
    --dataset_config_path dataloader/dataset_diff.yml \
    --flare_config_path dataloader/flare_config.yml \
    --tracker_run_name lucid_train \
    --tracker_project_name lucid \
    --pretrained_model_name_or_path stabilityai/sd-turbo \
    --flare_disentanglement_path ./checkpoints/flare_disentanglement/latest.pth \
    --output_dir ./lucid_checkpoints \
    --lambda_lpips 1.0 \
    --lambda_l2 1.0 \
    --neg_prob 0.2 \
    --flare_reinput_prob 0.5 \
    --lambda_intrinsic_multilevel 1.0 \
    --eval_freq 500 \
    --num_samples_eval 100 \
    --viz_freq 100 \
    --lora_rank_vae 4 \
    --timestep 199 \
    --resolution 512 \
    --train_batch_size 4 \
    --num_training_epochs 1000 \
    --max_train_steps 100000 \
    --checkpointing_steps 1000 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-6 \
    --lr_scheduler linear \
    --lr_warmup_steps 500 \
    --dataloader_num_workers 0 \
    --max_grad_norm 1.0 \
    --report_to wandb \
    --mixed_precision fp16 \
    --ms_unet
