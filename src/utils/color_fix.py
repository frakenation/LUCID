import torch
import torch.nn.functional as F
import numpy as np


def wavelet_blur(image, radius):
    """
    Apply wavelet blur to the input tensor
    
    Args:
        image: Input tensor [B, C, H, W]
        radius: Blur radius (dilation factor)
    
    Returns:
        Blurred tensor [B, C, H, W]
    """
    original_dtype = image.dtype
    
    if image.dtype == torch.bfloat16:
        image = image.float()
    
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None]
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    
    if original_dtype == torch.bfloat16:
        output = output.to(original_dtype)
    
    return output


def wavelet_decomposition(image, levels=5):
    """
    Apply wavelet decomposition to the input tensor
    
    Args:
        image: Input tensor [B, C, H, W]
        levels: Number of decomposition levels
    
    Returns:
        high_freq: High frequency component [B, C, H, W]
        low_freq: Low frequency component [B, C, H, W]
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq
    return high_freq, low_freq


def wavelet_color_fix(content_feat, style_feat):
    """
    Apply wavelet color fix: preserve content's high frequency,
    use style's low frequency for color alignment
    
    Args:
        content_feat: The image to fix (e.g., base_result), [B, 3, H, W]
        style_feat: The reference image (e.g., lq - flare), [B, 3, H, W]
    
    Returns:
        Color-fixed image [B, 3, H, W]
    """
    original_dtype = content_feat.dtype
    
    if content_feat.dtype == torch.bfloat16:
        content_feat = content_feat.float()
        style_feat = style_feat.float()
    
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    
    result = content_high_freq + style_low_freq
    
    if original_dtype == torch.bfloat16:
        result = result.to(original_dtype)
    
    return result


def morphological_erosion(mask, kernel_size=3):
    """
    Apply morphological erosion using min pooling
    
    Args:
        mask: Binary mask [B, C, H, W]
        kernel_size: Erosion kernel size
    
    Returns:
        Eroded mask [B, C, H, W]
    """
    padding = kernel_size // 2
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
    return eroded


def morphological_dilation(mask, kernel_size=3):
    """
    Apply morphological dilation using max pooling
    
    Args:
        mask: Binary mask [B, C, H, W]
        kernel_size: Dilation kernel size
    
    Returns:
        Dilated mask [B, C, H, W]
    """
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
    return dilated


def get_largest_connected_component(mask):
    """
    Extract the largest connected component from a binary mask
    
    Args:
        mask: Binary mask [B, 1, H, W], values in {0, 1}
    
    Returns:
        Largest connected component mask [B, 1, H, W]
    """
    try:
        import cv2
    except ImportError:
        print("Warning: cv2 not available, skipping connected component analysis")
        return mask
    
    batch_size = mask.shape[0]
    device = mask.device
    dtype = mask.dtype
    
    result_masks = []
    
    for b in range(batch_size):
        mask_np = mask[b, 0].cpu().numpy()
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )
        
        if num_labels <= 1:
            result_masks.append(torch.zeros_like(mask[b:b+1]))
            continue
        
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        
        largest_cc = (labels == largest_label).astype(np.float32)
        largest_cc_tensor = torch.from_numpy(largest_cc).unsqueeze(0).unsqueeze(0).to(device).to(dtype)
        result_masks.append(largest_cc_tensor)
    
    return torch.cat(result_masks, dim=0)


def gaussian_blur_mask(mask, kernel_size=15, sigma=5.0):
    """
    Apply Gaussian blur to smooth mask boundaries
    
    Args:
        mask: Input mask [B, C, H, W]
        kernel_size: Gaussian kernel size (must be odd)
        sigma: Gaussian sigma
    
    Returns:
        Blurred mask [B, C, H, W]
    """
    original_dtype = mask.dtype
    
    if mask.dtype == torch.bfloat16:
        mask = mask.float()
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    channels = mask.shape[1]
    
    x_coord = torch.arange(kernel_size).float() - kernel_size // 2
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    
    gaussian_kernel = torch.exp(-torch.sum(xy_grid ** 2., dim=-1) / (2 * sigma ** 2.))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_kernel = gaussian_kernel.to(mask.device).to(mask.dtype)
    
    padding = kernel_size // 2
    blurred = F.conv2d(mask, gaussian_kernel, padding=padding, groups=channels)
    
    if original_dtype == torch.bfloat16:
        blurred = blurred.to(original_dtype)
    
    return blurred


def refine_flare_mask(
    flare,
    mask_threshold=0.05,
    erosion_kernel=3,
    dilation_kernel=5,
    use_largest_cc=True,
    blur_kernel=15,
    blur_sigma=5.0,
    min_area_ratio=0.0001
):
    """
    Refine flare mask with morphological operations and smoothing
    
    Pipeline:
    1. Create initial binary mask from flare intensity
    2. Morphological opening (erosion + dilation) to remove noise
    3. Extract largest connected component
    4. Gaussian blur for smooth boundaries
    
    Args:
        flare: Flare component [B, 3, H, W]
        mask_threshold: Initial threshold for mask creation
        erosion_kernel: Kernel size for erosion (opening operation)
        dilation_kernel: Kernel size for dilation (opening + final dilation)
        use_largest_cc: Whether to keep only largest connected component
        blur_kernel: Kernel size for Gaussian blur
        blur_sigma: Sigma for Gaussian blur
        min_area_ratio: Minimum area ratio to keep a component
    
    Returns:
        Refined mask [B, 1, H, W], values in [0, 1] after blur
    """
    flare_intensity = torch.mean(flare, dim=1, keepdim=True)
    mask = (flare_intensity > mask_threshold).float()
    
    if erosion_kernel > 0:
        mask = morphological_erosion(mask, erosion_kernel)
    
    if dilation_kernel > 0:
        mask = morphological_dilation(mask, dilation_kernel)
    
    if use_largest_cc:
        H, W = mask.shape[2], mask.shape[3]
        total_area = H * W
        min_area = int(total_area * min_area_ratio)
        
        try:
            mask = get_largest_connected_component(mask)
            
            mask_area = mask.sum().item()
            if mask_area < min_area:
                print(f"Warning: Largest CC area {mask_area} < min_area {min_area}, using original mask")
        except Exception as e:
            print(f"Warning: Connected component analysis failed: {e}, using morphological mask")
    
    if blur_kernel > 0 and blur_sigma > 0:
        mask = gaussian_blur_mask(mask, blur_kernel, blur_sigma)
        mask = torch.clamp(mask, 0, 1)
    
    return mask