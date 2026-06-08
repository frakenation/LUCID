"""Flare synthesis primitives shared by LUCID dataloaders."""

from dataclasses import dataclass
import random

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk
from torch.distributions import Normal

from .common import load_image_any_format


class RandomGammaCorrection(object):
    """Torchvision-compatible random or fixed gamma correction."""

    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, image):
        if self.gamma is None:
            gammas = [0.5, 1, 2]
            self.gamma = random.choice(gammas)
            return TF.adjust_gamma(image, self.gamma, gain=1)
        if isinstance(self.gamma, tuple):
            gamma = random.uniform(*self.gamma)
            return TF.adjust_gamma(image, gamma, gain=1)
        if self.gamma == 0:
            return image
        return TF.adjust_gamma(image, self.gamma, gain=1)


class TranslationTransform(object):
    """Translate a tensor by a fixed pixel offset."""

    def __init__(self, position):
        self.position = position

    def __call__(self, x):
        return TF.affine(x, angle=0, scale=1, shear=[0, 0], translate=list(self.position))


@dataclass
class LowLightSample:
    """Low-light tensors and transforms sampled once per synthetic flare example."""

    image: object
    tensor_gamma: torch.Tensor
    tensor_noisy: torch.Tensor
    flare_dc_offset: float
    light_pos: list
    adjust_gamma: RandomGammaCorrection
    adjust_gamma_reverse: RandomGammaCorrection
    color_jitter: transforms.ColorJitter
    blur_transform: transforms.GaussianBlur


def preprocess_pil_image(image, resize_target, crop_size, target_size=None):
    """Resize by short side, center-crop, convert to tensor, and optionally resize output tensor."""
    image = resize_short_side(image, resize_target)
    image = transforms.CenterCrop(crop_size)(image)
    image_tensor = transforms.ToTensor()(image)
    if target_size is not None and target_size != (crop_size, crop_size):
        image_tensor = transforms.Resize(target_size)(image_tensor)
    return image_tensor


def center_crop_square(image):
    """Center-crop a PIL image to its shortest side."""
    w, h = image.size
    return transforms.CenterCrop(min(w, h))(image)


def resize_to_match(image_a, image_b):
    """Resize two PIL images to a shared square size before tensor merging."""
    w_a, _ = image_a.size
    w_b, _ = image_b.size
    min_side = min(w_a, w_b)
    resize_op = transforms.Resize((min_side, min_side))
    if min_side == w_a:
        image_b = resize_op(image_b)
    else:
        image_a = resize_op(image_a)
    return image_a, image_b


def resize_short_side(img, target_size):
    """Resize a PIL image so the short side equals target_size."""
    w, h = img.size

    if w < h:
        scale = target_size / w
        new_w = target_size
        new_h = int(h * scale)
    else:
        scale = target_size / h
        new_h = target_size
        new_w = int(w * scale)

    return img.resize((new_w, new_h), Image.LANCZOS)


def resize_tensor_if_needed(tensor, source_size, target_size):
    """Resize a tensor only when the configured output size differs."""
    if target_size == (source_size, source_size):
        return tensor
    return transforms.Resize(target_size)(tensor)


def resize_tensor_dict_if_needed(tensors, source_size, target_size):
    """Resize all tensors in a mapping when the output size differs from the crop size."""
    if target_size == (source_size, source_size):
        return tensors
    resize_transform = transforms.Resize(target_size)
    return {key: resize_transform(value) for key, value in tensors.items()}


def build_color_jitter(config):
    """Create the configured color jitter transform."""
    color_config = config['color_jitter']
    return transforms.ColorJitter(
        brightness=tuple(color_config['brightness']),
        hue=color_config['hue']
    )


def build_blur_transform(config):
    """Create the configured flare blur transform."""
    blur_config = config['blur']
    return transforms.GaussianBlur(
        blur_config['kernel_size'],
        sigma=(blur_config['sigma_min'], blur_config['sigma_max'])
    )


def build_flare_transform(config, light_pos, image_size):
    """Create the randomized flare placement transform for one sample."""
    transform_config = config['flare_transform']
    return transforms.Compose([
        transforms.RandomHorizontalFlip() if transform_config['random_flip'] else transforms.Lambda(lambda x: x),
        transforms.RandomVerticalFlip() if transform_config['random_flip'] else transforms.Lambda(lambda x: x),
        transforms.RandomAffine(
            degrees=tuple(transform_config['rotation_degrees']),
            scale=tuple(transform_config['scale_range']),
            translate=tuple(transform_config['translate']),
            shear=tuple(transform_config['shear_range'])
        ),
        TranslationTransform(light_pos),
        transforms.CenterCrop((image_size, image_size)),
    ])


def sample_gamma_pair(config):
    """Sample forward and reverse gamma correction transforms."""
    gamma_config = config['gamma']
    gamma = np.random.uniform(gamma_config['min'], gamma_config['max'])
    return gamma, RandomGammaCorrection(gamma), RandomGammaCorrection(1 / gamma)


def add_low_light_noise(lol_tensor, config):
    """Apply configured low-light noise, gain, and return DC flare offset."""
    noise_config = config['noise']
    sigma_chi = noise_config['sigma_chi_factor'] * np.random.chisquare(df=1)
    lol_tensor = Normal(lol_tensor, sigma_chi).sample()
    gain = np.random.uniform(noise_config['gain_min'], noise_config['gain_max'])
    flare_dc_offset = np.random.uniform(noise_config['dc_offset_min'], noise_config['dc_offset_max'])
    lol_tensor = torch.clamp(gain * lol_tensor, min=0, max=1)
    return lol_tensor, flare_dc_offset


def detect_relative_light_pos(lol_tensor, gamma, config, image_size):
    """Detect light position and convert it to center-relative translation."""
    light_config = config['light_detection']
    light_pos = plot_light_pos(lol_tensor, light_config['threshold_factor'] * gamma)
    return [light_pos[0] - image_size // 2, light_pos[1] - image_size // 2]


def plot_light_pos(input_img, threshold):
    """Estimate the dominant light-source location from a luminance threshold."""
    luminance = 0.3 * input_img[0] + 0.59 * input_img[1] + 0.11 * input_img[2]
    luminance_mask = luminance > threshold
    luminance_mask_np = luminance_mask.numpy()
    struc = disk(3)
    img_e = ndimage.binary_erosion(luminance_mask_np, structure=struc)
    img_ed = ndimage.binary_dilation(img_e, structure=struc)
    labels = label(img_ed)

    if labels.max() == 0:
        return (255, 255)

    largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    largest_cc = largest_cc.astype(int)
    properties = regionprops(largest_cc, largest_cc)
    weighted_center_of_mass = properties[0].weighted_centroid
    return (int(weighted_center_of_mass[1]), int(weighted_center_of_mass[0]))


def remove_background(image):
    """Remove the constant background level from a flare asset."""
    image = np.float32(np.array(image))
    eps = 1e-7
    rgb_max = np.max(image, (0, 1))
    rgb_min = np.min(image, (0, 1))
    image = (image - rgb_min) * rgb_max / (rgb_max - rgb_min + eps)
    return torch.from_numpy(image)


def load_centered_flare_tensor(flare_path, to_tensor, adjust_gamma):
    """Load a flare asset, center-crop it to square, convert to tensor, and apply gamma."""
    flare_img = center_crop_square(load_image_any_format(flare_path))
    flare_tensor = to_tensor(flare_img)
    return adjust_gamma(flare_tensor)


def sample_low_light_state(lol_img, config, image_size):
    """Prepare the gamma-space low-light tensor and per-sample flare transform context."""
    to_tensor = transforms.ToTensor()
    gamma, adjust_gamma, adjust_gamma_reverse = sample_gamma_pair(config)
    color_jitter = build_color_jitter(config)
    blur_transform = build_blur_transform(config)

    lol_tensor = to_tensor(lol_img)
    lol_tensor_gamma = adjust_gamma(lol_tensor)
    lol_tensor_noisy, flare_dc_offset = add_low_light_noise(lol_tensor_gamma, config)
    light_pos = detect_relative_light_pos(lol_tensor_noisy, gamma, config, image_size)

    return LowLightSample(
        image=lol_img,
        tensor_gamma=lol_tensor_gamma,
        tensor_noisy=lol_tensor_noisy,
        flare_dc_offset=flare_dc_offset,
        light_pos=light_pos,
        adjust_gamma=adjust_gamma,
        adjust_gamma_reverse=adjust_gamma_reverse,
        color_jitter=color_jitter,
        blur_transform=blur_transform,
    )


def sample_flare_tensor(scattering_flare_list, reflective_flare_list, reflective_enabled, to_tensor, adjust_gamma):
    """Sample and combine scattering and optional reflective flare assets."""
    scattering_path = random.choice(scattering_flare_list)
    scattering_img = center_crop_square(load_image_any_format(scattering_path))

    if not reflective_enabled:
        return to_tensor(scattering_img), scattering_path

    reflective_path = random.choice(reflective_flare_list)
    reflective_img = load_image_any_format(reflective_path)
    scattering_img, reflective_img = resize_to_match(scattering_img, reflective_img)

    scattering_tensor = adjust_gamma(to_tensor(scattering_img))
    reflective_tensor = adjust_gamma(to_tensor(reflective_img))
    flare_tensor = torch.clamp(scattering_tensor + reflective_tensor, min=0, max=1)
    return flare_tensor, scattering_path


def synthesize_flare(flare_tensor, config, low_light_state, image_size, transform_params=None):
    """Transform a flare tensor into gamma-space flare aligned with the low-light sample."""
    if transform_params is None:
        transform_flare = build_flare_transform(config, low_light_state.light_pos, image_size)
        flare_tensor = remove_background(flare_tensor)
        flare_tensor = transform_flare(flare_tensor)
    else:
        flare_tensor = apply_deterministic_transform(
            remove_background(flare_tensor),
            transform_params,
            image_size,
            low_light_state.light_pos,
        )

    flare_tensor = low_light_state.color_jitter(flare_tensor)
    flare_tensor = low_light_state.blur_transform(flare_tensor)
    flare_tensor = flare_tensor + low_light_state.flare_dc_offset
    return torch.clamp(flare_tensor, min=0, max=1)


def add_flare_to_low_light(flare_tensor, low_light_tensor):
    """Add gamma-space flare to a gamma-space low-light tensor."""
    return torch.clamp(flare_tensor + low_light_tensor, min=0, max=1)


def generate_transform_params(transform_config):
    """Sample one deterministic transform parameter set for paired flare assets."""
    h_flip = random.random() < 0.5 if transform_config['random_flip'] else False
    v_flip = random.random() < 0.5 if transform_config['random_flip'] else False

    angle = random.uniform(*transform_config['rotation_degrees'])
    scale = random.uniform(*transform_config['scale_range'])

    translate_range = transform_config['translate']
    if isinstance(translate_range, (list, tuple)) and len(translate_range) == 2:
        translate_x = random.uniform(*translate_range)
        translate_y = random.uniform(*translate_range)
        translate = (translate_x, translate_y)
    else:
        translate = tuple(translate_range)

    shear_range = transform_config['shear_range']
    shear = (random.uniform(*shear_range), 0)

    return {
        'h_flip': h_flip,
        'v_flip': v_flip,
        'angle': angle,
        'scale': scale,
        'translate': translate,
        'shear': shear
    }


def apply_deterministic_transform(flare_tensor, transform_params, image_size, light_pos):
    """Apply a sampled transform parameter set to a flare tensor."""
    if transform_params['h_flip']:
        flare_tensor = TF.hflip(flare_tensor)
    if transform_params['v_flip']:
        flare_tensor = TF.vflip(flare_tensor)

    flare_tensor = TF.affine(
        flare_tensor,
        angle=transform_params['angle'],
        translate=transform_params['translate'],
        scale=transform_params['scale'],
        shear=transform_params['shear']
    )

    flare_tensor = TF.affine(flare_tensor, angle=0, scale=1, shear=[0, 0], translate=list(light_pos))
    return transforms.CenterCrop((image_size, image_size))(flare_tensor)
