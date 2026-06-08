import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import cv2

from .model_cfg import LUCIDRestorationModel
from dataloader.paired_datasets import LUCIDPairedDataset
from .HDRgen_pyramid import HDRFusion

def srgb_to_linear(img):
    """Convert sRGB to linear RGB"""
    return torch.where(
        img <= 0.04045,
        img / 12.92,
        torch.pow((img + 0.055) / 1.055, 2.4)
    )

def tensor_to_numpy_srgb(tensor):
    """Convert tensor to numpy in sRGB space without linearization."""
    tensor = tensor.cpu()
    if tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)

    img_np = tensor.permute(1, 2, 0).numpy().astype(np.float32)
    return img_np

def calculate_ev_diff(input_img, output_img):
    """Calculate EV difference"""
    output_img = (output_img + 1.0) / 2.0
    output_img = torch.clamp(output_img, 0, 1)

    input_linear = srgb_to_linear(input_img)
    output_linear = srgb_to_linear(output_img)

    input_luminance = 0.2126 * input_linear[0] + 0.7152 * input_linear[1] + 0.0722 * input_linear[2]
    output_luminance = 0.2126 * output_linear[0] + 0.7152 * output_linear[1] + 0.0722 * output_linear[2]

    input_median = torch.median(input_luminance).item()
    output_median = torch.median(output_luminance).item()

    input_median = max(input_median, 1e-6)
    output_median = max(output_median, 1e-6)

    ev_diff = np.log2(output_median / input_median)
    return ev_diff

def save_tensor_as_image(tensor, path):
    """Save tensor as image"""
    tensor = tensor.cpu()
    if tensor.min() < 0:
         tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(path)
    return image

def tensor_to_numpy_linear(tensor):
    """Convert tensor to numpy in linear space"""
    tensor = tensor.cpu()
    if tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)

    tensor_linear = srgb_to_linear(tensor)
    img_np = tensor_linear.permute(1, 2, 0).numpy().astype(np.float32)
    return img_np

def tonemap_hdr(hdr_image, method='drago'):
    """Tone mapping using OpenCV"""
    hdr_image = hdr_image.astype(np.float32)

    if method == 'drago':
        tonemap = cv2.createTonemapDrago(gamma=2.2, saturation=1.0, bias=0.85)
    elif method == 'mantiuk':
        tonemap = cv2.createTonemapMantiuk(gamma=2.2, scale=0.7, saturation=1.0)
    elif method == 'reinhard':
        tonemap = cv2.createTonemapReinhard(gamma=2.2, intensity=0.0, light_adapt=0.8, color_adapt=0.0)
    else:
        tonemap = cv2.createTonemap(gamma=2.2)

    hdr_bgr = cv2.cvtColor(hdr_image, cv2.COLOR_RGB2BGR)
    ldr_bgr = tonemap.process(hdr_bgr)
    ldr_rgb = cv2.cvtColor(ldr_bgr, cv2.COLOR_BGR2RGB)

    ldr_rgb = np.clip(ldr_rgb, 0, 1)
    return ldr_rgb

def main():
    """Run LUCID CFG-sequence inference and optional HDR fusion."""
    parser = argparse.ArgumentParser(description='LUCID inference with CFG scales and HDR fusion')

    parser.add_argument('--dataset_config_path', type=str, required=True)

    parser.add_argument('--pretrained_model_name_or_path', type=str, default='stabilityai/sd-turbo')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--flare_disentanglement_path', type=str, default=None)

    parser.add_argument('--lora_rank_vae', type=int, default=4)
    parser.add_argument('--timestep', type=int, default=999)
    parser.add_argument('--ms_unet', action='store_true')

    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--positive_prompt', type=str, default=" ")
    parser.add_argument('--negative_prompt', type=str, default="")

    parser.add_argument('--cfg_scales', type=str, required=True)

    parser.add_argument('--enable_hdr_fusion', action='store_true')
    parser.add_argument('--hdr_pyramid_levels', type=int, default=5)
    parser.add_argument('--hdr_saturation_threshold', type=float, default=0.98)
    parser.add_argument('--hdr_darkness_threshold', type=float, default=0.02)
    parser.add_argument('--visualize_weights', action='store_true')
    parser.add_argument('--tonemap_method', type=str, default='drago',
                        choices=['drago', 'mantiuk', 'reinhard'],
                        help='Tone mapping method for HDR visualization')

    parser.add_argument('--output_dir', type=str, default='./inference_results')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    if not args.flare_disentanglement_path:
        raise ValueError(
            "LUCID inference requires --flare_disentanglement_path for the flare "
            "disentanglement checkpoint."
        )

    try:
        cfg_scales = [float(x.strip()) for x in args.cfg_scales.split(',')]
        if not cfg_scales:
            raise ValueError("No CFG scales provided")
    except Exception as e:
        raise ValueError(f"Error parsing cfg_scales: {e}") from e

    if args.enable_hdr_fusion and len(cfg_scales) < 2:
        args.enable_hdr_fusion = False

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

    hdr_fusion = None
    if args.enable_hdr_fusion:
        hdr_fusion = HDRFusion(
            saturation_threshold=args.hdr_saturation_threshold,
            darkness_threshold=args.hdr_darkness_threshold,
            pyramid_levels=args.hdr_pyramid_levels,
            visualize_weights=args.visualize_weights
        )

    ev_stats = {cfg: [] for cfg in cfg_scales}
    image_ev_records = {}

    sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
            if args.num_samples is not None and sample_idx >= args.num_samples:
                break

            lq_images = batch['input_image'].to(args.device)
            positive_prompt = args.positive_prompt

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
                    current_basename = f'image_{sample_idx:04d}'

                image_folder = os.path.join(args.output_dir, current_basename)
                os.makedirs(image_folder, exist_ok=True)

                lq_image_single = lq_images[i:i+1]
                input_img_cpu = lq_images[i].cpu()

                enhanced_images_srgb = []
                ev_values = []

                image_ev_records[current_basename] = {}

                for cfg_scale in cfg_scales:
                    result = model(
                        lq_image=lq_image_single,
                        gt_image=None,
                        neg_mode=False,
                        positive_prompt=positive_prompt,
                        negative_prompt=args.negative_prompt,
                        enable_cfg=True,
                        cfg_scale=cfg_scale
                    )

                    enhanced_image = result['enhanced_image'][0]

                    ev_diff = calculate_ev_diff(input_img_cpu, enhanced_image.cpu())
                    ev_stats[cfg_scale].append(ev_diff)
                    image_ev_records[current_basename][f'cfg_{cfg_scale}'] = ev_diff
                    ev_values.append(ev_diff)

                    output_filename = f'cfg_{cfg_scale}.png'
                    output_path = os.path.join(image_folder, output_filename)
                    save_tensor_as_image(enhanced_image, output_path)

                    if args.enable_hdr_fusion:
                        img_srgb = tensor_to_numpy_srgb(enhanced_image)
                        enhanced_images_srgb.append(img_srgb)

                if args.enable_hdr_fusion and len(enhanced_images_srgb) >= 2:
                    hdr_output_path = os.path.join(image_folder, 'fused_hdr.exr')
                    vis_folder = os.path.join(image_folder, 'hdr_visualization') if \
                                args.visualize_weights else None

                    if vis_folder:
                        os.makedirs(vis_folder, exist_ok=True)

                    hdr_result = hdr_fusion.fuse(
                        images_srgb=enhanced_images_srgb,
                        cfg_values=cfg_scales,
                        output_path=hdr_output_path,
                        vis_folder=vis_folder
                    )

                    if hdr_result is not None:
                        tonemapped = tonemap_hdr(hdr_result, method=args.tonemap_method)
                        tonemapped_uint8 = (tonemapped * 255).astype(np.uint8)
                        tonemapped_img = Image.fromarray(tonemapped_uint8)
                        tonemapped_path = os.path.join(args.output_dir, f'{current_basename}_tonemapped.png')
                        tonemapped_img.save(tonemapped_path)

                sample_idx += 1

    stats_summary = {}
    for cfg_scale in cfg_scales:
        ev_values = ev_stats[cfg_scale]
        if ev_values:
            mean_ev = np.mean(ev_values)
            std_ev = np.std(ev_values)

            stats_summary[f'cfg_{cfg_scale}'] = {
                'mean': float(mean_ev),
                'std': float(std_ev),
                'median': float(np.median(ev_values)),
                'min': float(np.min(ev_values)),
                'max': float(np.max(ev_values))
            }

    stats_file = os.path.join(args.output_dir, 'ev_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'summary': stats_summary,
            'per_image': image_ev_records
        }, f, indent=2)

if __name__ == "__main__":
    main()
