import os
import gc
import lpips
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

from .model_cfg import LUCIDRestorationModel, load_ckpt_from_state_dict, save_ckpt
from dataloader.paired_datasets import LUCIDFlareReinputDataset, LUCIDPairedDataset

def tensor_to_wandb_image(tensor, input_range='01'):
    if input_range == 'neg11':
        tensor = (tensor + 1.0) / 2.0
    elif input_range != '01':
        raise ValueError(f"Unsupported input range: {input_range}")

    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    if tensor.dim() == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor.permute(1, 2, 0).cpu().numpy()
    else:
        raise ValueError(f"Unexpected tensor dimensions: {tensor.shape}")

def apply_gamma_augmentation(image, gamma_min, gamma_max):
    gamma = np.random.uniform(gamma_min, gamma_max)
    image_transformed = torch.pow(image.clamp(min=1e-8), 1.0 / gamma)
    return image_transformed.clamp(0, 1.0)

def main(args):
    if not args.flare_disentanglement_path:
        raise ValueError(
            "LUCID training requires a flare disentanglement checkpoint. "
            "Use --flare_disentanglement_path."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.output_dir,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints_no_intrinsic"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    colorfix_config = None
    if args.enable_colorfix:
        colorfix_config = {
            'mask_threshold': args.colorfix_mask_threshold,
            'erosion_kernel': args.colorfix_erosion_kernel,
            'dilation_kernel': args.colorfix_dilation_kernel,
            'use_largest_cc': args.colorfix_use_largest_cc,
            'blur_kernel': args.colorfix_blur_kernel,
            'blur_sigma': args.colorfix_blur_sigma,
            'min_area_ratio': args.colorfix_min_area_ratio
        }
        print("Color Fix Configuration:")
        for k, v in colorfix_config.items():
            print(f"  {k}: {v}")

    net_lucid = LUCIDRestorationModel(
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
        ms_unet=args.ms_unet,
        flare_disentanglement_path=args.flare_disentanglement_path,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        enable_colorfix=args.enable_colorfix,
        colorfix_config=colorfix_config
    )
    net_lucid.set_train()

    net_lucid.vae.decoder.skip_enabled = [True, True, True, True]

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_lucid.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_lucid.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    net_vgg = torchvision.models.vgg16(pretrained=True).features
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    layers_to_opt = []
    layers_to_opt += list(net_lucid.unet.parameters())

    for n, _p in net_lucid.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    layers_to_opt = layers_to_opt + list(net_lucid.vae.decoder.skip_conv_1.parameters()) + \
        list(net_lucid.vae.decoder.skip_conv_2.parameters()) + \
        list(net_lucid.vae.decoder.skip_conv_3.parameters()) + \
        list(net_lucid.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    ds_train = LUCIDPairedDataset(
        args.dataset_config_path,
        height=args.resolution,
        width=args.resolution,
        flare_config_path=args.flare_config_path,
        require_lq=True,
    )

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers
    )

    dl_flare_reinput = None
    flare_dataloader_iter = None
    if args.flare_reinput_prob > 0:
        flare_dataset_config = args.flare_dataset_config_path or args.dataset_config_path

        ds_flare_reinput = LUCIDFlareReinputDataset(
            flare_dataset_config,
            height=args.resolution,
            width=args.resolution,
            flare_config_path=args.flare_config_path
        )

        dl_flare_reinput = torch.utils.data.DataLoader(
            ds_flare_reinput,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers
        )

        dl_flare_reinput = accelerator.prepare(dl_flare_reinput)
        flare_dataloader_iter = iter(dl_flare_reinput)
        print(f"Flare reinput mode enabled with probability {args.flare_reinput_prob}")
        print(f"Using dataset config: {flare_dataset_config}")

    dataset_val = LUCIDPairedDataset(
        dataset_config_path=args.dataset_config_path,
        flare_config_path=args.flare_config_path,
        height=args.resolution,
        width=args.resolution,
        require_lq=True,
    )
    random.Random(42).shuffle(dataset_val.lol_data_list)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    global_step = 0
    if args.resume is not None:
        if os.path.isdir(args.resume):

            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")))
            print(f"Loading checkpoint from {ckpt_files[-1]}")
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_lucid, optimizer = load_ckpt_from_state_dict(
                net_lucid, optimizer, ckpt_files[-1]
            )
        elif args.resume.endswith(".pkl"):
            print(f"Loading checkpoint from {args.resume}")
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_lucid, optimizer = load_ckpt_from_state_dict(
                net_lucid, optimizer, args.resume
            )
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("Training from scratch")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_lucid.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)

    net_lucid, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_lucid, optimizer, dl_train, lr_scheduler
    )

    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)
    t_vgg_renorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if accelerator.is_main_process:
        init_kwargs = {
            "wandb": {
                "name": args.tracker_run_name,
                "dir": args.output_dir,
            },
        }
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):

            use_flare_reinput = False
            if args.flare_reinput_prob > 0 and np.random.rand() < args.flare_reinput_prob:
                use_flare_reinput = True
                try:
                    batch = next(flare_dataloader_iter)
                except StopIteration:
                    flare_dataloader_iter = iter(dl_flare_reinput)
                    batch = next(flare_dataloader_iter)

            l_acc = [net_lucid]
            with accelerator.accumulate(*l_acc):
                lq_image = batch["input_image"]
                positive_prompt = batch["positive_prompt"]
                negative_prompt = batch["negative_prompt"]
                B = lq_image.shape[0]

                if use_flare_reinput:
                    gt_image = batch["gt_image"]
                    lol_gt = batch.get("lol_gt", None)
                    use_neg_mode_flare = np.random.rand() < args.neg_prob
                    if use_neg_mode_flare and lol_gt is not None:
                        result = net_lucid(
                            lq_image,
                            gt_image=gt_image,
                            lol_gt=lol_gt,
                            neg_mode=True,
                            negative_prompt=negative_prompt
                        )
                    else:
                        result = net_lucid(
                            lq_image,
                            gt_image=gt_image,
                            lol_gt=None,
                            neg_mode=False,
                            positive_prompt=positive_prompt
                        )
                else:
                    gt_image = batch["gt_image"]
                    use_neg_mode = np.random.rand() < args.neg_prob
                    lol_gt = batch.get("lol_gt", None)

                    if use_neg_mode and lol_gt is not None:
                        lol_gt = lol_gt.to(dtype=weight_dtype)
                        result = net_lucid(
                            lq_image,
                            gt_image=gt_image,
                            lol_gt=lol_gt,
                            neg_mode=True,
                            negative_prompt=negative_prompt
                        )
                    else:
                        result = net_lucid(
                            lq_image,
                            gt_image=gt_image,
                            lol_gt=None,
                            neg_mode=False,
                            positive_prompt=positive_prompt
                        )

                enhanced_image = result['enhanced_image']
                target_image = result['target_image']
                flare = result['flare']

                if args.gamma_augmentation:
                    target_image = apply_gamma_augmentation(
                        target_image,
                        args.gamma_range_min,
                        args.gamma_range_max
                    )

                normalize = transforms.Normalize([0.5], [0.5])
                target_image_neg11 = normalize(target_image)

                loss_l2 = F.mse_loss(enhanced_image.float(), target_image_neg11.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(enhanced_image.float(), target_image_neg11.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips

                if args.lambda_intrinsic_multilevel > 0:
                    loss_intrinsic_multilevel = net_lucid.compute_multi_level_intrinsic_loss(
                        enhanced_image.float(), target_image_neg11.float()
                    ) * args.lambda_intrinsic_multilevel
                    loss += loss_intrinsic_multilevel
                else:
                    loss_intrinsic_multilevel = torch.tensor(0.0).to(weight_dtype)

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss_intrinsic_multilevel"] = loss_intrinsic_multilevel.detach().item()
                    logs["loss"] = loss.detach().item()

                    if use_flare_reinput:
                        logs["mode"] = "flare_reinput"
                        logs["flare_type"] = batch.get("flare_type", "unknown")
                    elif 'use_neg_mode' in locals() and use_neg_mode:
                        logs["mode"] = "negative"
                    else:
                        logs["mode"] = "normal"

                    progress_bar.set_postfix(**logs)

                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/lq_image": [wandb.Image(tensor_to_wandb_image(lq_image[idx], input_range='01'), caption=f"idx={idx}") for idx in range(B)],
                            "train/enhanced": [wandb.Image(tensor_to_wandb_image(enhanced_image[idx], input_range='neg11'), caption=f"idx={idx}") for idx in range(B)],
                            "train/gt_image": [wandb.Image(tensor_to_wandb_image(target_image[idx], input_range='01'), caption=f"idx={idx}") for idx in range(B)],
                            "train/flare": [wandb.Image(tensor_to_wandb_image(flare[idx], input_range='01'), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                        if use_flare_reinput:
                            log_dict["train/flare_size"] = batch.get("flare_type", "unknown")
                            log_dict["train/positive_prompt"] = positive_prompt[0] if isinstance(positive_prompt, list) else positive_prompt

                        for k in log_dict:
                            logs[k] = log_dict[k]

                    if use_flare_reinput:
                        logs["mode"] = "flare_reinput"
                        if 'use_neg_mode_flare' in locals() and use_neg_mode_flare:
                            logs["mode"] = "flare_reinput_neg"
                        logs["flare_type"] = batch.get("flare_type", "unknown")

                    if global_step % args.checkpointing_steps == 0:
                        outf = os.path.join(args.output_dir, "checkpoints_no_intrinsic", f"model_{global_step}.pkl")
                        save_ckpt(accelerator.unwrap_model(net_lucid), optimizer, outf)
                        print(f"\nCheckpoint saved at {outf}")

                    if global_step % args.eval_freq == 0:
                        l_l2 = []
                        l_lpips = []
                        l_intrinsic_multilevel = []

                        with torch.no_grad():
                            for idx, batch_val in enumerate(dl_val):
                                if idx >= args.num_samples_eval:
                                    break

                                lq_image_val = batch_val["input_image"].to(accelerator.device)
                                gt_image_val = batch_val["gt_image"].to(accelerator.device)
                                positive_prompt_val = batch_val["positive_prompt"]

                                result_val = accelerator.unwrap_model(net_lucid)(
                                    lq_image_val,
                                    gt_image=gt_image_val,
                                    lol_gt=None,
                                    neg_mode=False,
                                    positive_prompt=positive_prompt_val
                                )

                                enhanced_image_val = result_val['enhanced_image']
                                normalize = transforms.Normalize([0.5], [0.5])
                                gt_image_val_neg11 = normalize(gt_image_val)

                                loss_l2_val = F.mse_loss(enhanced_image_val.float(), gt_image_val_neg11.float(), reduction="mean")
                                loss_lpips_val = net_lpips(enhanced_image_val.float(), gt_image_val_neg11.float()).mean()

                                l_l2.append(loss_l2_val.item())
                                l_lpips.append(loss_lpips_val.item())

                                if args.lambda_intrinsic_multilevel > 0:
                                    loss_intrinsic_multilevel_val = accelerator.unwrap_model(net_lucid).compute_multi_level_intrinsic_loss(
                                        enhanced_image_val.float(), gt_image_val_neg11.float()
                                    )
                                    l_intrinsic_multilevel.append(loss_intrinsic_multilevel_val.item())

                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        if args.lambda_intrinsic_multilevel > 0 and len(l_intrinsic_multilevel) > 0:
                            logs["val/intrinsic_multilevel"] = np.mean(l_intrinsic_multilevel)

                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--enable_colorfix", action="store_true",
                       help="Enable wavelet color fix in preprocessing pipeline")
    parser.add_argument("--colorfix_mask_threshold", type=float, default=0.125,
                       help="Threshold for creating flare mask [0.01-0.2]")
    parser.add_argument("--colorfix_erosion_kernel", type=int, default=7,
                       help="Kernel size for mask erosion in opening operation [0-7]")
    parser.add_argument("--colorfix_dilation_kernel", type=int, default=9,
                       help="Kernel size for mask dilation [0-9]")
    parser.add_argument("--colorfix_use_largest_cc", action="store_true",
                       help="Use largest connected component for mask refinement")
    parser.add_argument("--colorfix_blur_kernel", type=int, default=15,
                       help="Kernel size for Gaussian blur of mask [0-31]")
    parser.add_argument("--colorfix_blur_sigma", type=float, default=5.0,
                       help="Sigma for Gaussian blur of mask [0-10]")
    parser.add_argument("--colorfix_min_area_ratio", type=float, default=0.0001,
                       help="Minimum area ratio for connected component [0-0.01]")

    parser.add_argument("--neg_prob", type=float, default=0.1,
                   help="Probability of using negative mode during training (default: 0.1)")
    parser.add_argument("--flare_reinput_prob", type=float, default=0.0,
                   help="Probability of using flare reinput mode during training (default: 0.0)")
    parser.add_argument("--flare_dataset_config_path", type=str, default=None,
                    help="Path to flare dataset config YAML file for reinput mode. If not provided, uses --dataset_config_path")

    parser.add_argument("--lambda_intrinsic_multilevel", default=0.0, type=float,
                       help="Weight for multi-level intrinsic consistency loss")

    parser.add_argument("--gamma_augmentation", action="store_true",
                    help="Enable gamma augmentation on GT images during training")
    parser.add_argument("--gamma_range_min", default=0.8, type=float,
                    help="Minimum gamma value for augmentation")
    parser.add_argument("--gamma_range_max", default=1.2, type=float,
                    help="Maximum gamma value for augmentation")

    parser.add_argument("--dataset_config_path", required=True, type=str, help="Path to dataset config YAML file")
    parser.add_argument("--flare_config_path", required=True, type=str, help="Path to flare config YAML file")

    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="lucid", help="The name of the wandb project to log to.")
    parser.add_argument("--tracker_run_name", type=str, required=True)

    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--ms_unet", action="store_true", help="Use Mixing-State UNet")

    parser.add_argument("--flare_disentanglement_path", type=str, help="Path to pretrained flare disentanglement checkpoint")

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=1000)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--checkpointing_steps", type=int, default=1000,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="linear",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()

    main(args)
