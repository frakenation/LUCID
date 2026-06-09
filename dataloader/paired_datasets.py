"""Dataset definitions for LUCID restoration and flare-synthesis training."""

import os
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from dataloader.utils.common import (
    IMAGE_EXTENSIONS,
    basename_without_ext,
    load_aligned_restoration_mappings,
    load_config,
    load_flare_assets,
    load_image_any_format,
    load_lol_gt_pairs,
)
from dataloader.utils.flare_synthesis import (
    add_flare_to_low_light,
    generate_transform_params,
    load_centered_flare_tensor,
    preprocess_pil_image,
    resize_short_side,
    resize_tensor_dict_if_needed,
    resize_tensor_if_needed,
    sample_flare_tensor,
    sample_low_light_state,
    synthesize_flare,
)
from dataloader.utils.prompts import (
    FLARE_REINPUT_TYPE_WEIGHTS,
    FLARE_TYPE_PROMPT_PREFIXES,
    NEGATIVE_PROMPTS,
    RESTORATION_POSITIVE_PROMPTS,
    build_prompt_bank,
    prefixed_prompts,
    sample_prompt_pair,
)


class FlareDisentanglementDataset(data.Dataset):
    """Synthetic flare decomposition samples for training the disentanglement network."""

    def __init__(self, dataset_config_path, height=512, width=512, flare_config_path='flare_config.yml'):
        super().__init__()

        self.target_size = (height, width)
        self.ext = IMAGE_EXTENSIONS
        self.config = load_config(flare_config_path)
        self.image_size = self.config['image_size']['final_crop']

        dataset_config = load_config(dataset_config_path)
        self.lol_data_list, self.gt_data_dict = load_lol_gt_pairs(dataset_config, self.ext)

        flare_assets = load_flare_assets(dataset_config, self.ext)
        self.scattering_flare_list = flare_assets['scattering']
        self.reflective_flare_list = flare_assets['reflective']
        self.reflective_flag = flare_assets['reflective_enabled']

        print("FlareDisentanglementDataset loaded:")
        print(f"  LOL images: {len(self.lol_data_list)}")
        print(f"  GT mappings: {len(self.gt_data_dict)}")
        print(f"  Scattering flare images: {len(self.scattering_flare_list)}")
        print(f"  Reflective flare images: {len(self.reflective_flare_list)}")
        print(f"  Reflective flare enabled: {self.reflective_flag}")

        if len(self.scattering_flare_list) == 0:
            raise ValueError("No scattering flare images found.")

    def __len__(self):
        return len(self.lol_data_list)

    def __getitem__(self, index):
        """Return one synthetic flare decomposition sample."""
        lol_path = self.lol_data_list[index]
        lol_img = self._load_training_image(lol_path)
        gt_img = self._load_optional_gt(lol_path)

        low_light = sample_low_light_state(lol_img, self.config, self.image_size)
        to_tensor = transforms.ToTensor()
        flare_tensor, _ = sample_flare_tensor(
            self.scattering_flare_list,
            self.reflective_flare_list,
            self.reflective_flag,
            to_tensor,
            low_light.adjust_gamma,
        )

        flare_gamma = synthesize_flare(flare_tensor, self.config, low_light, self.image_size)
        lq_gamma = add_flare_to_low_light(flare_gamma, low_light.tensor_noisy)

        outputs = resize_tensor_dict_if_needed(
            {
                "lol": low_light.adjust_gamma_reverse(low_light.tensor_noisy),
                "flare": low_light.adjust_gamma_reverse(flare_gamma),
                "lq": low_light.adjust_gamma_reverse(lq_gamma),
            },
            self.image_size,
            self.target_size,
        )

        result = {
            "input_image": outputs["lq"],
            "lol_gt": outputs["lol"],
            "flare_gt": outputs["flare"],
            "has_gt": True,
            "data_type": "synthetic_flare",
        }

        if gt_img is not None:
            gt_tensor = to_tensor(gt_img)
            result["gt_image"] = resize_tensor_if_needed(gt_tensor, self.image_size, self.target_size)

        return result

    def _load_training_image(self, image_path):
        image = load_image_any_format(image_path)
        image = resize_short_side(image, self.config['image_size']['resize_target'])
        return transforms.CenterCrop(self.image_size)(image)

    def _load_optional_gt(self, lol_path):
        if lol_path not in self.gt_data_dict:
            return None
        return self._load_training_image(self.gt_data_dict[lol_path])


class LUCIDPairedDataset(data.Dataset):
    """Aligned restoration pairs for the main LUCID restoration model."""

    def __init__(self, dataset_config_path=None, height=512, width=512, flare_config_path=None,
                 require_lq=False, input_dir=None):
        super().__init__()

        self.target_size = (height, width)
        self.ext = IMAGE_EXTENSIONS
        self.require_lq = require_lq
        self.flare_config = None
        if flare_config_path is not None:
            self.flare_config = load_config(flare_config_path)
            self.image_size = self.flare_config['image_size']['final_crop']
            self.resize_target = self.flare_config['image_size']['resize_target']
        else:
            self.image_size = min(height, width)
            self.resize_target = min(height, width)

        if input_dir is not None:
            dataset_config = {'datasets': [{'lq_image_path': input_dir}]}
        elif dataset_config_path is not None:
            dataset_config = load_config(dataset_config_path)
        else:
            raise ValueError("Provide dataset_config_path or input_dir.")
        self.data_mappings = load_aligned_restoration_mappings(
            dataset_config,
            self.ext,
            require_lq=self.require_lq,
        )
        if self.require_lq and not self.data_mappings:
            raise ValueError(
                "No aligned GT/LQ pairs found. Diffusion training requires lq_image_path "
                "and matching GT/LQ basenames."
            )
        if not self.data_mappings:
            raise ValueError("No input images found.")
        self.lol_data_list = [
            mapping['lq_image_path'] or mapping['lol_gt_path']
            for mapping in self.data_mappings
            if mapping['lq_image_path'] or mapping['lol_gt_path']
        ]

        self.positive_prompts = prefixed_prompts("no flare", RESTORATION_POSITIVE_PROMPTS)
        self.negative_prompts = NEGATIVE_PROMPTS

        print("LUCIDPairedDataset loaded:")
        print(f"  Valid data mappings: {len(self.data_mappings)}")
        print(f"  LOL data list (for compatibility): {len(self.lol_data_list)}")
        print(f"  Target size: {self.target_size}")
        print(f"  Intermediate crop size: {self.image_size}")

    def preprocess_image_consistent(self, image):
        """Apply the restoration pipeline resize/crop/tensor path."""
        return preprocess_pil_image(image, self.resize_target, self.image_size, self.target_size)

    def load_datasets(self, dataset_config):
        """Compatibility hook for callers that expect this method to populate mappings."""
        self.data_mappings.extend(
            load_aligned_restoration_mappings(dataset_config, self.ext, require_lq=self.require_lq)
        )

    def __len__(self):
        return len(self.data_mappings)

    def __getitem__(self, idx):
        """Return one aligned restoration sample."""
        mapping = self.data_mappings[idx]

        try:
            input_tensor = self._load_input_tensor(mapping)
            if input_tensor is None:
                return self.__getitem__((idx + 1) % len(self.data_mappings))

            result = {
                "input_image": input_tensor,
                "data_type": "preprocessed_or_lol",
                "basename": mapping['basename'],
            }

            gt_tensor = self._load_gt_tensor(mapping)
            result["has_gt"] = gt_tensor is not None
            if gt_tensor is not None:
                result["gt_image"] = gt_tensor

            lol_gt_tensor = self._load_lol_gt_tensor(mapping)
            if lol_gt_tensor is not None:
                result["lol_gt"] = lol_gt_tensor

            positive_prompt, negative_prompt = sample_prompt_pair(self.positive_prompts, self.negative_prompts)
            result["positive_prompt"] = positive_prompt
            result["negative_prompt"] = negative_prompt

            return result

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self.data_mappings))

    def _load_gt_tensor(self, mapping):
        gt_path = mapping.get('gt_image_path')
        if not gt_path or not os.path.exists(gt_path):
            return None
        gt_image = load_image_any_format(gt_path)
        return self.preprocess_image_consistent(gt_image)

    def _load_input_tensor(self, mapping):
        lq_path = mapping.get('lq_image_path')
        if lq_path and os.path.exists(lq_path):
            input_image = load_image_any_format(lq_path)
            input_tensor = transforms.ToTensor()(input_image)
            if input_tensor.shape[-2:] != self.target_size:
                input_tensor = self.preprocess_image_consistent(input_image)
            return input_tensor

        if self.require_lq:
            return None

        lol_path = mapping.get('lol_gt_path')
        if lol_path and os.path.exists(lol_path):
            input_image = load_image_any_format(lol_path)
            return self.preprocess_image_consistent(input_image)

        return None

    def _load_lol_gt_tensor(self, mapping):
        lol_path = mapping.get('lol_gt_path')
        if not lol_path or not os.path.exists(lol_path):
            return None
        lol_gt_image = load_image_any_format(lol_path)
        return self.preprocess_image_consistent(lol_gt_image)


class LUCIDFlareReinputDataset(data.Dataset):
    """Synthetic restoration samples that reinsert flare into GT and low-light targets."""

    def __init__(self, dataset_config_path, height=512, width=512, flare_config_path='flare_config.yml'):
        super().__init__()

        self.target_size = (height, width)
        self.ext = IMAGE_EXTENSIONS
        self.config = load_config(flare_config_path)
        self.image_size = self.config['image_size']['final_crop']

        dataset_config = load_config(dataset_config_path)
        self.lol_data_list, self.gt_data_dict = load_lol_gt_pairs(dataset_config, self.ext)

        flare_assets = load_flare_assets(dataset_config, self.ext, include_light_sources=True)
        self.scattering_flare_list = flare_assets['scattering']
        self.light_source_list = flare_assets['light_sources']
        self.light_source_dict = flare_assets['light_source_by_basename']
        self.reflective_flare_list = flare_assets['reflective']
        self.reflective_flag = flare_assets['reflective_enabled']

        self.prompt_bank = build_prompt_bank(
            FLARE_TYPE_PROMPT_PREFIXES,
            RESTORATION_POSITIVE_PROMPTS,
            NEGATIVE_PROMPTS,
        )
        self.positive_prompt_templates_whole_flare = self.prompt_bank['whole_flare']['positive']
        self.positive_prompt_templates_light_source = self.prompt_bank['light_source']['positive']
        self.negative_prompt_templates_whole_flare = self.prompt_bank['whole_flare']['negative']
        self.negative_prompt_templates_light_source = self.prompt_bank['light_source']['negative']

        print("LUCIDFlareReinputDataset loaded:")
        print(f"  LOL images: {len(self.lol_data_list)}")
        print(f"  GT mappings: {len(self.gt_data_dict)}")
        print(f"  Scattering flare images: {len(self.scattering_flare_list)}")
        print(f"  Light-source flare images: {len(self.light_source_list)}")
        print(f"  Reflective flare images: {len(self.reflective_flare_list)}")

        if len(self.scattering_flare_list) == 0:
            raise ValueError("No scattering flare images found.")
        if FLARE_REINPUT_TYPE_WEIGHTS.get('light_source', 0.0) > 0 and len(self.light_source_list) == 0:
            raise ValueError("Light-source flare reinput is enabled, but no light-source images were found.")

    def __len__(self):
        return len(self.lol_data_list)

    def __getitem__(self, index):
        """Return one flare-reinput restoration sample."""
        lol_path = self.lol_data_list[index]
        if lol_path not in self.gt_data_dict:
            return self.__getitem__((index + 1) % len(self.lol_data_list))

        lol_img = self._load_training_image(lol_path)
        gt_img = self._load_training_image(self.gt_data_dict[lol_path])

        low_light = sample_low_light_state(lol_img, self.config, self.image_size)
        flare_type = self._sample_flare_type()

        input_flare_gamma, gt_flare_gamma, lol_flare_gamma = self._sample_reinput_flare_triplet(
            flare_type,
            low_light,
        )
        lq_gamma = add_flare_to_low_light(input_flare_gamma, low_light.tensor_noisy)

        to_tensor = transforms.ToTensor()
        gt_tensor_gamma = low_light.adjust_gamma(to_tensor(gt_img))
        outputs = resize_tensor_dict_if_needed(
            {
                "lq": low_light.adjust_gamma_reverse(lq_gamma),
                "gt_with_flare": low_light.adjust_gamma_reverse(gt_flare_gamma + gt_tensor_gamma),
                "lol_with_flare": low_light.adjust_gamma_reverse(lol_flare_gamma + low_light.tensor_gamma),
                "gt_flare": low_light.adjust_gamma_reverse(gt_flare_gamma),
                "lol_flare": low_light.adjust_gamma_reverse(lol_flare_gamma),
            },
            self.image_size,
            self.target_size,
        )

        prompt_templates = self.prompt_bank[flare_type]
        positive_prompt, negative_prompt = sample_prompt_pair(
            prompt_templates["positive"],
            prompt_templates["negative"],
        )

        return {
            "input_image": outputs["lq"],
            "gt_image": torch.clamp(outputs["gt_with_flare"], 0, 1),
            "lol_gt": torch.clamp(outputs["lol_with_flare"], 0, 1),
            "gt_flare": outputs["gt_flare"],
            "lol_flare": outputs["lol_flare"],
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "has_gt": True,
            "data_type": "synthetic_flare_with_scale",
            "flare_type": flare_type,
        }

    def _load_training_image(self, image_path):
        image = load_image_any_format(image_path)
        image = resize_short_side(image, self.config['image_size']['resize_target'])
        return transforms.CenterCrop(self.image_size)(image)

    def _sample_flare_type(self):
        flare_types = list(FLARE_REINPUT_TYPE_WEIGHTS.keys())
        return random.choices(
            flare_types,
            weights=[FLARE_REINPUT_TYPE_WEIGHTS[key] for key in flare_types],
            k=1,
        )[0]

    def _sample_reinput_flare_triplet(self, flare_type, low_light):
        to_tensor = transforms.ToTensor()
        transform_params = generate_transform_params(self.config['flare_transform'])
        scattering_flare_path = random.choice(self.scattering_flare_list)
        scattering_basename = basename_without_ext(scattering_flare_path)
        scattering_flare_tensor = load_centered_flare_tensor(
            scattering_flare_path,
            to_tensor,
            low_light.adjust_gamma,
        )
        input_flare = synthesize_flare(
            scattering_flare_tensor,
            self.config,
            low_light,
            self.image_size,
            transform_params,
        )

        if flare_type == 'whole_flare':
            return input_flare, input_flare, input_flare

        light_source_path = self.light_source_dict.get(scattering_basename)
        if light_source_path is None:
            light_source_path = random.choice(self.light_source_list)

        light_source_tensor = load_centered_flare_tensor(light_source_path, to_tensor, low_light.adjust_gamma)
        gt_flare = synthesize_flare(light_source_tensor, self.config, low_light, self.image_size, transform_params)

        lol_transform_params = transform_params.copy()
        lol_transform_params['scale'] = transform_params['scale'] * 0.5
        lol_flare = synthesize_flare(
            light_source_tensor,
            self.config,
            low_light,
            self.image_size,
            lol_transform_params,
        )

        return input_flare, gt_flare, lol_flare


if __name__ == "__main__":
    print("LUCID datasets ready to use.")
