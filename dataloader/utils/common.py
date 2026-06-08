"""Common file, config, and image-loading utilities for LUCID datasets."""

import glob
import os

from PIL import Image
import yaml

IMAGE_EXTENSIONS = ['png', 'jpeg', 'jpg', 'bmp', 'tif', 'arw', 'raw', 'nef', 'cr2', 'dng']


def load_config(config_path):
    """Load a YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def as_list(paths):
    """Normalize optional path config values into a list."""
    if paths is None:
        return []
    if isinstance(paths, str):
        return [paths]
    return list(paths)


def list_image_files(folder, extensions=IMAGE_EXTENSIONS):
    """List supported image files in one folder with case-insensitive extensions."""
    if not folder:
        return []

    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f'*.{ext}')))
        files.extend(glob.glob(os.path.join(folder, f'*.{ext.upper()}')))
    return files


def basename_without_ext(path):
    """Return the filename stem used for image alignment."""
    return os.path.splitext(os.path.basename(path))[0]


def index_by_basename(paths):
    """Build a basename-to-path mapping."""
    return {basename_without_ext(path): path for path in paths}


def resolve_dataset_subdir(root_path, dataset_info):
    """Resolve optional per-dataset subfolder layout."""
    dataset_name = dataset_info.get('name')
    if dataset_name:
        dataset_path = os.path.join(root_path, dataset_name)
        if os.path.exists(dataset_path):
            return dataset_path
    return root_path


def load_lol_gt_pairs(dataset_config, extensions=IMAGE_EXTENSIONS):
    """Load low-light image list and optional GT mapping keyed by low-light path."""
    lol_data_list = []
    gt_data_dict = {}

    for dataset_info in dataset_config['datasets']:
        lol_gt_path = dataset_info['lol_gt_path']
        gt_image_path = dataset_info.get('gt_image_path')
        current_lol_list = list_image_files(lol_gt_path, extensions)

        if gt_image_path:
            gt_dict = index_by_basename(list_image_files(gt_image_path, extensions))
            for lol_path in current_lol_list:
                gt_path = gt_dict.get(basename_without_ext(lol_path))
                if gt_path:
                    gt_data_dict[lol_path] = gt_path

        lol_data_list.extend(current_lol_list)

    return lol_data_list, gt_data_dict


def load_flare_assets(dataset_config, extensions=IMAGE_EXTENSIONS, include_light_sources=False):
    """Load flare asset paths from dataset config."""
    flare_config = dataset_config.get('flare_datasets', dataset_config)
    assets = {
        'scattering': [],
        'reflective': [],
        'reflective_enabled': False,
        'light_sources': [],
        'light_source_by_basename': {},
    }

    for flare_path in as_list(flare_config.get('scattering_flare_paths', [])):
        assets['scattering'].extend(list_image_files(flare_path, extensions))

    reflective_paths = as_list(flare_config.get('reflective_flare_paths'))
    assets['reflective_enabled'] = len(reflective_paths) > 0
    for flare_path in reflective_paths:
        assets['reflective'].extend(list_image_files(flare_path, extensions))

    if include_light_sources:
        for light_path in as_list(flare_config.get('light_source_paths', [])):
            assets['light_sources'].extend(list_image_files(light_path, extensions))
        assets['light_source_by_basename'] = index_by_basename(assets['light_sources'])

    return assets


def load_aligned_restoration_mappings(dataset_config, extensions=IMAGE_EXTENSIONS):
    """Create LUCID restoration mappings aligned by GT basename."""
    data_mappings = []

    for dataset_info in dataset_config['datasets']:
        lol_gt_path = dataset_info.get('lol_gt_path')
        gt_image_path = dataset_info.get('gt_image_path')
        lq_image_path = dataset_info.get('lq_image_path')
        if not gt_image_path:
            print("Skipping dataset without gt_image_path")
            continue

        gt_image_list = list_image_files(gt_image_path, extensions)
        if len(gt_image_list) == 0:
            print(f"No GT images found in {gt_image_path}")
            continue

        lol_gt_dict = index_by_basename(list_image_files(lol_gt_path, extensions)) if lol_gt_path else {}
        lq_image_dict = {}
        if lq_image_path:
            search_path = resolve_dataset_subdir(lq_image_path, dataset_info)
            lq_image_dict = index_by_basename(list_image_files(search_path, extensions))

        for gt_path in gt_image_list:
            basename = basename_without_ext(gt_path)
            data_mappings.append({
                'gt_image_path': gt_path,
                'lol_gt_path': lol_gt_dict.get(basename),
                'lq_image_path': lq_image_dict.get(basename),
                'basename': basename
            })

    return data_mappings


def load_image_any_format(img_path):
    """Load RGB images and common camera RAW files as PIL RGB images."""
    ext = os.path.splitext(img_path)[1].lower()

    if ext in ['.arw', '.raw', '.nef', '.cr2', '.dng']:
        try:
            import rawpy
            with rawpy.imread(img_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=False,
                    output_bps=8
                )
            return Image.fromarray(rgb)
        except Exception as e:
            print(f"Error loading RAW file {img_path}: {e}")
            raise

    return Image.open(img_path).convert('RGB')

