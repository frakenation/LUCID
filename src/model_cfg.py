import os
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
os.environ['DISABLE_BITSANDBYTES'] = '1'
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from peft import LoraConfig
from einops import rearrange, repeat
from .Flare_Disentangle import FlareDisentanglementNetwork
from .utils.checkpoint import (
    load_flare_state,
    load_lucid_checkpoint,
    load_lucid_components,
    load_lucid_state,
    save_lucid_state,
)

from .utils.color_fix import (
    wavelet_color_fix,
    refine_flare_mask,
    morphological_dilation
)

def make_1step_sched(model_path="stabilityai/sd-turbo"):
    noise_scheduler_1step = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step

def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []

    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)

    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample

def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]

        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)

            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)

    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample

def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")

def load_ckpt_from_state_dict(net_lucid, optimizer, pretrained_path):
    return load_lucid_state(net_lucid, optimizer, pretrained_path)

def save_ckpt(net_lucid, optimizer, outf):
    save_lucid_state(net_lucid, optimizer, outf)

class LUCIDRestorationModel(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints",
                 lora_rank_vae=4, ms_unet=False, timestep=999, flare_disentanglement_path=None,
                 pretrained_model_name_or_path="stabilityai/sd-turbo", enable_colorfix=True, colorfix_config=None):
        super().__init__()

        base_model_path = pretrained_model_name_or_path or "stabilityai/sd-turbo"

        print(f"Loading base model from: {base_model_path}")

        if base_model_path != "stabilityai/sd-turbo" and not os.path.exists(base_model_path):
            print(f"Warning: Local path {base_model_path} does not exist, falling back to stabilityai/sd-turbo")
            base_model_path = "stabilityai/sd-turbo"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder").cuda()
            self.sched = make_1step_sched(base_model_path)
        except Exception as e:
            print(f"Error loading from {base_model_path}: {e}")
            print("Falling back to online download...")
            base_model_path = "stabilityai/sd-turbo"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder").cuda()
            self.sched = make_1step_sched(base_model_path)

        vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False

        if flare_disentanglement_path is not None:
            self.flare_disentanglement_net = FlareDisentanglementNetwork()
            load_flare_state(self.flare_disentanglement_net, flare_disentanglement_path)
            self.flare_disentanglement_net.eval()
            print(f"Loaded flare disentanglement network from {flare_disentanglement_path}")
        else:
            self.flare_disentanglement_net = None
            print("No flare disentanglement checkpoint provided; using development fallback")

        if ms_unet:
            from .mixing_state import UNet2DConditionModel
        else:
            from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")

        if pretrained_path is not None:
            sd = load_lucid_checkpoint(pretrained_path)
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            load_lucid_components(vae, unet, sd)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            target_modules_vae = []

            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]

            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                    target_modules.append(name)
            target_modules_vae = target_modules
            vae.encoder.requires_grad_(False)

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae

        unet.to("cuda")
        vae.to("cuda")
        if self.flare_disentanglement_net is not None:
            self.flare_disentanglement_net.to("cuda")

        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 0.2
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        print("="*50)
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        if self.flare_disentanglement_net is not None:
            print(f"Number of parameters in flare disentanglement network (frozen): {sum(p.numel() for p in self.flare_disentanglement_net.parameters()) / 1e6:.2f}M")
        print("="*50)

        self.enable_colorfix = enable_colorfix
        self.colorfix_config = colorfix_config or {
            'mask_threshold': 0.05,
            'erosion_kernel': 3,
            'dilation_kernel': 5,
            'use_largest_cc': True,
            'blur_kernel': 15,
            'blur_sigma': 5.0,
            'min_area_ratio': 0.0001
        }

        if self.enable_colorfix:
            print("Color fix enabled with config:", self.colorfix_config)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        if self.flare_disentanglement_net is not None:
            self.flare_disentanglement_net.eval()

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

        if self.flare_disentanglement_net is not None:
            self.flare_disentanglement_net.eval()
            for param in self.flare_disentanglement_net.parameters():
                param.requires_grad = False

    def _encode_prompt(self, prompt, batch_size):
        """Encode text prompt to embeddings"""
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size

        caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                        padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
        caption_enc = self.text_encoder(caption_tokens)[0]
        return caption_enc

    def background_with_noise_injection(self, lq_image, background_context, background, flare,
                                        flare_factor=0.8, min_context=0.01, brightness_boost=1.0,
                                        base_noise_std=0.02, flare_noise_std=0.05,
                                        dilation_kernel_size=7):

        flare_luminance = (0.299 * flare[:, 0:1, :, :] +
                           0.587 * flare[:, 1:2, :, :] +
                           0.114 * flare[:, 2:3, :, :])

        clean_context = background_context - flare_factor * flare_luminance
        clean_context = torch.clamp(clean_context, min_context, 1.0)

        if brightness_boost != 1.0:
            clean_context = torch.pow(clean_context, 1.0 / brightness_boost)
            clean_context = torch.clamp(clean_context, min_context, 1.0)

        base_result = torch.clamp(background, 0, 1)

        refined_mask = refine_flare_mask(
            flare=flare,
            mask_threshold=self.colorfix_config['mask_threshold'],
            erosion_kernel=self.colorfix_config['erosion_kernel'],
            dilation_kernel=self.colorfix_config['dilation_kernel'],
            use_largest_cc=self.colorfix_config['use_largest_cc'],
            blur_kernel=self.colorfix_config['blur_kernel'],
            blur_sigma=self.colorfix_config['blur_sigma'],
            min_area_ratio=self.colorfix_config.get('min_area_ratio', 0.0001)
        )

        if self.enable_colorfix:
            refined_mask_3ch = refined_mask.repeat(1, 3, 1, 1)
            reference = lq_image * (1.0 - refined_mask_3ch)

            base_result_fixed = wavelet_color_fix(base_result, reference)
            base_result_fixed = torch.clamp(base_result_fixed, 0, 1)
        else:
            base_result_fixed = base_result

        dilated_noise_mask = morphological_dilation(refined_mask, dilation_kernel_size)

        base_noise = torch.randn_like(base_result_fixed) * base_noise_std
        flare_noise = torch.randn_like(base_result_fixed) * flare_noise_std

        dilated_noise_mask_3ch = dilated_noise_mask.repeat(1, 3, 1, 1)
        non_flare_mask_3ch = 1.0 - dilated_noise_mask_3ch

        noise = base_noise * non_flare_mask_3ch + flare_noise * dilated_noise_mask_3ch

        noisy_result = base_result_fixed + noise

        return torch.clamp(noisy_result, 0, 1)

    def compute_multi_level_intrinsic_loss(self, enhanced_image, gt_image):

        if self.flare_disentanglement_net is None:
            return torch.tensor(0.0).to(enhanced_image.device)

        disentanglement_dtype = next(self.flare_disentanglement_net.parameters()).dtype

        enhanced_norm = ((enhanced_image + 1.0) / 2.0).to(dtype=disentanglement_dtype)
        gt_norm = ((gt_image + 1.0) / 2.0).to(dtype=disentanglement_dtype)
        enhanced_norm = torch.clamp(enhanced_norm, 0, 1)
        gt_norm = torch.clamp(gt_norm, 0, 1)

        with torch.no_grad():

            enhanced_features = self.flare_disentanglement_net.extract_features(enhanced_norm)
            gt_features = self.flare_disentanglement_net.extract_features(gt_norm)

        total_loss = torch.tensor(0.0, dtype=enhanced_image.dtype, device=enhanced_image.device)

        encoder_weights = enhanced_features['encoder_weights']
        for i, (enh_feat, gt_feat, weight) in enumerate(zip(
            enhanced_features['encoder_features'],
            gt_features['encoder_features'],
            encoder_weights
        )):
            if i < len(encoder_weights):
                encoder_loss = F.mse_loss(enh_feat, gt_feat)
                total_loss += weight * encoder_loss

        component_weights = enhanced_features['component_weights']

        flare_loss = F.mse_loss(
            enhanced_features['flare_feat'],
            gt_features['flare_feat']
        )
        total_loss += component_weights['flare_feat'] * flare_loss

        background_loss = F.mse_loss(
            enhanced_features['background_feat'],
            gt_features['background_feat']
        )
        total_loss += component_weights['background_feat'] * background_loss

        flare_att_loss = F.mse_loss(
            enhanced_features['flare_att'],
            gt_features['flare_att']
        )
        total_loss += component_weights['flare_att'] * flare_att_loss

        background_att_loss = F.mse_loss(
            enhanced_features['background_att'],
            gt_features['background_att']
        )
        total_loss += component_weights['background_att'] * background_att_loss

        refined_loss = F.mse_loss(
            enhanced_features['refined_feat'],
            gt_features['refined_feat']
        )
        total_loss += component_weights['refined_feat'] * refined_loss

        return total_loss

    def _forward_diffusion(self, x, caption_enc, enable_cfg=False, cfg_scale=1.5,
                      x_neg=None, neg_caption_enc=None):
        num_views = x.shape[1]
        batch_size = x.shape[0]
        x = rearrange(x, 'b v c h w -> (b v) c h w')

        z = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

        modified_skip_acts = []
        for skip_act in self.vae.encoder.current_down_blocks:
            new_skip_act = skip_act.clone()
            for b in range(batch_size):
                original_view_idx = b * num_views
                for v in range(num_views):
                    target_idx = b * num_views + v
                    new_skip_act[target_idx] = skip_act[original_view_idx]
            modified_skip_acts.append(new_skip_act)

        caption_enc_expanded = repeat(caption_enc, 'b n c -> (b v) n c', v=num_views)
        unet_input = z

        if enable_cfg:
            if x_neg is None or neg_caption_enc is None:
                raise ValueError("CFG mode requires both x_neg and neg_caption_enc")

            x_neg = rearrange(x_neg, 'b v c h w -> (b v) c h w')
            z_neg = self.vae.encode(x_neg).latent_dist.sample() * self.vae.config.scaling_factor

            neg_caption_enc_expanded = repeat(neg_caption_enc, 'b n c -> (b v) n c', v=num_views)

            model_pred_pos = self.unet(
                unet_input,
                self.timesteps,
                encoder_hidden_states=caption_enc_expanded
            ).sample
            z_denoised_pos = self.sched.step(
                model_pred_pos, self.timesteps, z, return_dict=True
            ).prev_sample

            model_pred_neg = self.unet(
                z_neg,
                self.timesteps,
                encoder_hidden_states=neg_caption_enc_expanded
            ).sample
            z_denoised_neg = self.sched.step(
                model_pred_neg, self.timesteps, z_neg, return_dict=True
            ).prev_sample

            z_denoised = z_denoised_neg + cfg_scale * (z_denoised_pos - z_denoised_neg)

        else:
            model_pred = self.unet(
                unet_input,
                self.timesteps,
                encoder_hidden_states=caption_enc_expanded
            ).sample
            z_denoised = self.sched.step(
                model_pred, self.timesteps, z, return_dict=True
            ).prev_sample

        self.vae.decoder.incoming_skip_acts = modified_skip_acts
        output_image = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views)

        return output_image

    def forward(self, lq_image, gt_image=None, lol_gt=None, neg_mode=False,
                positive_prompt="enhance this low-light image",
                negative_prompt="",
                timesteps=None, prompt_tokens=None,
                enable_cfg=False, cfg_scale=1.5):

        batch_size = lq_image.shape[0]

        if self.flare_disentanglement_net is not None:
            with torch.no_grad():
                background_context, background, flare, orth_loss = self.flare_disentanglement_net(lq_image)
        else:
            background_context = torch.mean(lq_image, dim=1, keepdim=True)
            background = lq_image
            flare = torch.zeros_like(lq_image)
            orth_loss = torch.tensor(0.0).to(lq_image.device)

        processed_lq_image = self.background_with_noise_injection(
            lq_image, background_context, background, flare
        )

        normalize = transforms.Normalize([0.5], [0.5])

        if neg_mode:

            lq_image_neg11 = normalize(processed_lq_image)
            flare_neg11 = normalize(flare)
            positive_input = torch.stack([lq_image_neg11, flare_neg11], dim=1)
            target_image = lol_gt

            positive_caption_enc = self._encode_prompt(negative_prompt, batch_size)

            positive_output = self._forward_diffusion(
                positive_input,
                positive_caption_enc,
                enable_cfg=False
            )[:, 0]

        elif enable_cfg:

            lq_image_neg11 = normalize(processed_lq_image)
            lq_image_original_neg11 = normalize(lq_image)
            positive_input = torch.stack([lq_image_neg11, lq_image_original_neg11], dim=1)

            flare_neg11 = normalize(flare)
            negative_input = torch.stack([lq_image_neg11, flare_neg11], dim=1)

            target_image = gt_image

            positive_caption_enc = self._encode_prompt(positive_prompt, batch_size)
            neg_caption_enc = self._encode_prompt(negative_prompt, batch_size)

            positive_output = self._forward_diffusion(
                positive_input,
                positive_caption_enc,
                enable_cfg=True,
                cfg_scale=cfg_scale,
                x_neg=negative_input,
                neg_caption_enc=neg_caption_enc
            )[:, 0]

        else:

            lq_image_neg11 = normalize(processed_lq_image)
            lq_image_original_neg11 = normalize(lq_image)
            positive_input = torch.stack([lq_image_neg11, lq_image_original_neg11], dim=1)
            target_image = gt_image

            positive_caption_enc = self._encode_prompt(positive_prompt, batch_size)

            positive_output = self._forward_diffusion(
                positive_input,
                positive_caption_enc,
                enable_cfg=False
            )[:, 0]

        enhanced_image = positive_output

        return {
            'enhanced_image': enhanced_image,
            'background_context': background_context,
            'background': background,
            'flare': flare,
            'orth_loss': orth_loss,
            'processed_lq_image': processed_lq_image,
            'target_image': target_image
        }

    def sample(self, image, width, height, ref_image=None, timesteps=None, prompt=None, prompt_tokens=None):
        input_width, input_height = image.size
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)

        T = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
        ])
        if ref_image is None:
            x = T(image).unsqueeze(0).unsqueeze(0).cuda()
        else:
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            x = torch.stack([T(image), T(ref_image)], dim=0).unsqueeze(0).cuda()

        result = self.forward(x, positive_prompt=prompt)
        output_image = result['enhanced_image'][0]
        output_pil = transforms.ToPILImage()(output_image.cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)

        return output_pil

    def save_model(self, outf, optimizer):
        save_lucid_state(self, optimizer, outf, unet_key_filter=("lora", "conv_in"))
