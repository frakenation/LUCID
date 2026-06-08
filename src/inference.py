import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from .model_cfg import LUCIDRestorationModel
from dataloader.paired_datasets import LUCIDPairedDataset

def save_tensor_as_image(tensor, path):
    """Save a tensor as an image"""
    tensor = tensor.cpu()
    if tensor.min() < 0:
         tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(path)
    return image

def main():
    """Run LUCID single-CFG inference."""
    parser = argparse.ArgumentParser(description='LUCID single-CFG inference')

    parser.add_argument('--dataset_config_path', type=str, required=True,
                       help='Path to dataset configuration YAML file')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                       default='stabilityai/sd-turbo',
                       help='Base model path (SD-Turbo or local path)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to LUCID restoration checkpoint (.pkl file)')
    parser.add_argument('--flare_disentanglement_path', type=str, default=None,
                       help='Path to flare disentanglement checkpoint (.pth file)')

    parser.add_argument('--lora_rank_vae', type=int, default=4,
                       help='LoRA rank for VAE')
    parser.add_argument('--timestep', type=int, default=999,
                       help='Diffusion timestep')
    parser.add_argument('--ms_unet', action='store_true',
                       help='Use Mixing-State UNet')

    parser.add_argument('--resolution', type=int, default=512,
                       help='Resolution for processing')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to process (None for all)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of dataloader workers')

    parser.add_argument('--positive_prompt', type=str,
                       default=" ",
                       help='Positive prompt for enhancement')

    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Directory to save outputs')

    parser.add_argument("--inference_mode", type=str, default="normal",
                   choices=["normal", "negative", "cfg_guidance"],
                   help="Inference mode: normal (default enhancement), negative, or cfg_guidance (mixed)")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                   help="CFG guidance scale when using cfg_guidance mode (default: 1.5)")
    parser.add_argument('--negative_prompt', type=str, default="",
                   help='Negative prompt for CFG guidance')

    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()
    if not args.flare_disentanglement_path:
        raise ValueError(
            "LUCID inference requires --flare_disentanglement_path for the flare "
            "disentanglement checkpoint."
        )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = LUCIDPairedDataset(
        dataset_config_path=args.dataset_config_path,
        height=args.resolution,
        width=args.resolution
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    model = LUCIDRestorationModel(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        pretrained_path=args.model_path,
        flare_disentanglement_path=args.flare_disentanglement_path,
        timestep=args.timestep,
        ms_unet=args.ms_unet,
        lora_rank_vae=args.lora_rank_vae,
        enable_colorfix=True
    )
    model.set_eval()
    model = model.to(args.device)

    sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if args.num_samples is not None and sample_idx >= args.num_samples:
                break

            lq_images = batch['input_image'].to(args.device)

            positive_prompt = args.positive_prompt

            gt_images = batch.get('gt_image')
            if gt_images is not None:
                gt_images = gt_images.to(args.device)

            if args.inference_mode == "cfg_guidance":
                result = model(
                    lq_image=lq_images,
                    gt_image=None,
                    neg_mode=False,
                    positive_prompt=positive_prompt,
                    negative_prompt=args.negative_prompt,
                    enable_cfg=True,
                    cfg_scale=args.cfg_scale
                )
            elif args.inference_mode == "negative":
                result = model(
                    lq_image=lq_images,
                    gt_image=None,
                    neg_mode=True,
                    negative_prompt=args.negative_prompt,
                    enable_cfg=False
                )
            else:
                result = model(
                    lq_image=lq_images,
                    gt_image=None,
                    neg_mode=False,
                    positive_prompt=positive_prompt,
                    enable_cfg=False
                )

            enhanced_images = result['enhanced_image']

            for i in range(lq_images.shape[0]):
                if args.num_samples is not None and sample_idx >= args.num_samples:
                    break

                basename = batch.get('basename')
                if basename is not None:
                    if isinstance(basename, (list, tuple)):
                        current_basename = basename[i]
                    else:
                        current_basename = basename
                else:
                    current_basename = f'{sample_idx:04d}'

                enhanced_path = os.path.join(args.output_dir, f'{current_basename}.png')
                save_tensor_as_image(enhanced_images[i], enhanced_path)

                sample_idx += 1

    info_path = os.path.join(args.output_dir, 'inference_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"LUCID Inference Results\n")
        f.write(f"="*50 + "\n")
        f.write(f"Dataset config: {args.dataset_config_path}\n")
        f.write(f"Total samples processed: {sample_idx}\n")
        f.write(f"Model checkpoint: {args.model_path}\n")
        f.write(f"Flare disentanglement checkpoint: {args.flare_disentanglement_path}\n")
        f.write(f"Inference mode: {args.inference_mode}\n")
        if args.inference_mode == "cfg_guidance":
            f.write(f"CFG scale: {args.cfg_scale}\n")
        f.write(f"Timestep: {args.timestep}\n")
        f.write(f"Mixing-State UNet: {args.ms_unet}\n")
        f.write(f"Resolution: {args.resolution}x{args.resolution}\n")
        f.write(f"Positive prompt: {args.positive_prompt}\n")

if __name__ == "__main__":
    main()
