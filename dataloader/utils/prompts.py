"""Prompt templates used by LUCID restoration training datasets."""

import random

RESTORATION_POSITIVE_PROMPTS = [
    "enhance this low-light image with better illumination and clarity",
    "improve the lighting and remove noise from this dark image",
    "brighten this image while preserving natural colors and details",
    "restore this low-light photo to normal brightness levels",
    "enhance visibility and reduce artifacts in this dark image",
]

NEGATIVE_PROMPTS = [
    "add more noise and artifacts to this image",
    "make this image darker",
    "introduce more artifacts and reduce image quality",
    "add noise and degrade the image",
    "increase noise and lighting distortions",
]

FLARE_TYPE_PROMPT_PREFIXES = {
    "whole_flare": "whole flare",
    "light_source": "light source",
}

FLARE_REINPUT_TYPE_WEIGHTS = {
    "whole_flare": 0.0,
    "light_source": 1.0,
}


def prefixed_prompts(prefix, prompts):
    """Attach a conditioning prefix to each prompt template."""
    return [f"{prefix}, {prompt}" for prompt in prompts]


def sample_prompt_pair(positive_prompts, negative_prompts=NEGATIVE_PROMPTS):
    """Sample one positive/negative prompt pair without coupling template indices."""
    return random.choice(positive_prompts), random.choice(negative_prompts)


def build_prompt_bank(prefixes, positive_templates, negative_templates=NEGATIVE_PROMPTS):
    """Build positive and negative prompt banks keyed by conditioning type."""
    return {
        key: {
            "positive": prefixed_prompts(prefix, positive_templates),
            "negative": negative_templates,
        }
        for key, prefix in prefixes.items()
    }

