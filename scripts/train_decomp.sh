#!/usr/bin/env bash
set -e

export PYTHONPATH=.
python -m src.train_disentangle \
    --dataset_config_path dataloader/dataset_diff.yml \
    --flare_config_path dataloader/flare_config.yml \
    --epochs 200 \
    --batch_size 8 \
    --val_batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --min_lr 1e-7 \
    --grad_clip 1.0 \
    --lambda_supervised 1.0 \
    --lambda_physical 0.5 \
    --lambda_orthogonal 0.2 \
    --in_channels 3 \
    --base_channels 64 \
    --image_height 512 \
    --image_width 512 \
    --num_workers 4 \
    --save_interval 20 \
    --log_interval 10 \
    --val_interval 5 \
    --vis_interval 1500 \
    --checkpoint_dir ./checkpoints/flare_disentanglement \
    --device cuda
