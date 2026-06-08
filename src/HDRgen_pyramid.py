
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class HDRFusion:
    """
    HDR fusion using Laplacian Pyramid with gradient-based exposure estimation
    """

    def __init__(self, saturation_threshold=0.98, darkness_threshold=0.02,
                 pyramid_levels=5, visualize_weights=False):
        self.SATURATION = saturation_threshold
        self.DARKNESS = darkness_threshold
        self.PYRAMID_LEVELS = pyramid_levels
        self.visualize_weights = visualize_weights

        print(f"HDR Fusion initialized")
        print(f"  Saturation: {self.SATURATION}, Darkness: {self.DARKNESS}")
        print(f"  Pyramid levels: {self.PYRAMID_LEVELS}")
        print(f"  Visualize weights: {visualize_weights}")

    def srgb_to_linear(self, img_srgb):
        """Convert sRGB to linear RGB"""
        return np.where(
            img_srgb <= 0.04045,
            img_srgb / 12.92,
            np.power((img_srgb + 0.055) / 1.055, 2.4)
        )

    def estimate_exposure_gradient_based(self, images_linear, cfg_values):
        """
        Gradient-based exposure estimation
        """
        n_images = len(images_linear)
        ref_idx = len(cfg_values) // 2

        print(f"  Estimating relative exposures (gradient-based)...")
        print(f"  Reference: index {ref_idx} (CFG={cfg_values[ref_idx]})")

        gains = np.ones(n_images, dtype=np.float32)

        for i in range(n_images):
            if i == ref_idx:
                continue

            img_i = images_linear[i]
            img_ref = images_linear[ref_idx]

            gray_i = np.mean(img_i, axis=2)
            gray_ref = np.mean(img_ref, axis=2)

            grad_i_x = np.diff(gray_i, axis=1)
            grad_i_y = np.diff(gray_i, axis=0)
            grad_ref_x = np.diff(gray_ref, axis=1)
            grad_ref_y = np.diff(gray_ref, axis=0)

            threshold = 0.02
            mask_x = (np.abs(grad_ref_x) > threshold) & \
                     (np.abs(grad_i_x) > threshold) & \
                     (gray_ref[:, :-1] > 0.05) & (gray_ref[:, :-1] < 0.95) & \
                     (gray_i[:, :-1] > 0.05) & (gray_i[:, :-1] < 0.95)

            mask_y = (np.abs(grad_ref_y) > threshold) & \
                     (np.abs(grad_i_y) > threshold) & \
                     (gray_ref[:-1, :] > 0.05) & (gray_ref[:-1, :] < 0.95) & \
                     (gray_i[:-1, :] > 0.05) & (gray_i[:-1, :] < 0.95)

            valid_count_x = np.sum(mask_x)
            valid_count_y = np.sum(mask_y)

            if valid_count_x < 100 and valid_count_y < 100:
                print(f"  Warning: Image {i} insufficient gradients, using CFG fallback")
                gains[i] = cfg_values[i] / cfg_values[ref_idx]
                continue

            ratios = []
            if valid_count_x > 0:
                ratio_x = grad_i_x[mask_x] / (grad_ref_x[mask_x] + 1e-10)
                ratios.append(ratio_x)
            if valid_count_y > 0:
                ratio_y = grad_i_y[mask_y] / (grad_ref_y[mask_y] + 1e-10)
                ratios.append(ratio_y)

            all_ratios = np.concatenate([r.flatten() for r in ratios])

            q25, q75 = np.percentile(all_ratios, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            filtered_ratios = all_ratios[(all_ratios >= lower_bound) & (all_ratios <= upper_bound)]

            if len(filtered_ratios) > 0:
                gain = np.median(filtered_ratios)
                gains[i] = gain
                print(f"  Image {i} (CFG={cfg_values[i]}): gain={gain:.4f}")
            else:
                gains[i] = cfg_values[i] / cfg_values[ref_idx]
                print(f"  Warning: Image {i} fallback to CFG ratio")

        gains = gains / gains[ref_idx]
        print(f"  Final gains: {gains}")

        return gains

    def compute_quality_weight(self, srgb_img):
        """
        Compute quality weight in sRGB space (original method)
        """

        weight_exposure = 1.0 - np.abs(srgb_img - 0.5) * 2.0
        weight_exposure = np.clip(weight_exposure, 0, 1)

        mask_sat = (srgb_img >= self.SATURATION).astype(np.float32)
        mask_dark = (srgb_img <= self.DARKNESS).astype(np.float32)
        weight_valid = 1.0 - np.maximum(mask_sat, mask_dark)

        weight_final = weight_exposure * weight_valid
        weight_final = weight_final + 0.01

        return weight_final

    def compute_gradient_weight(self, img_linear):
        """
        Compute gradient-based weight map for visualization
        """
        gray = np.mean(img_linear, axis=2)

        grad_x = np.abs(np.diff(gray, axis=1))
        grad_y = np.abs(np.diff(gray, axis=0))

        grad_x_pad = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
        grad_y_pad = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')

        grad_magnitude = np.sqrt(grad_x_pad**2 + grad_y_pad**2)

        grad_weight = grad_magnitude / (np.max(grad_magnitude) + 1e-10)

        grad_weight = np.stack([grad_weight] * 3, axis=2)

        return grad_weight

    def _visualize_weights(self, images_srgb, images_linear, weights, cfg_values, vis_folder):
        """
        Visualize weight distribution - save each gradient weight map separately
        """
        print(f"  Generating weight visualizations...")

        n_images = len(images_srgb)

        for i in range(n_images):

            grad_weight = self.compute_gradient_weight(images_linear[i])
            grad_weight_gray = np.mean(grad_weight, axis=2)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            im = ax.imshow(grad_weight_gray, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Gradient Weight Map (CFG={cfg_values[i]})', fontsize=14)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

            plt.tight_layout()
            save_path = os.path.join(vis_folder, f'gradient_weight_cfg_{cfg_values[i]}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: gradient_weight_cfg_{cfg_values[i]}.png")

        for i in range(n_images):
            weight_gray = np.mean(weights[i], axis=2)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            im = ax.imshow(weight_gray, cmap='viridis', vmin=0, vmax=np.max(weight_gray))

            ax.axis('off')

            plt.tight_layout()
            save_path = os.path.join(vis_folder, f'quality_weight_cfg_{cfg_values[i]}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: quality_weight_cfg_{cfg_values[i]}.png")

        weight_sum = sum(weights)
        weights_normalized = [w / (weight_sum + 1e-10) for w in weights]

        for i in range(n_images):
            weight_gray = np.mean(weights_normalized[i], axis=2)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            im = ax.imshow(weight_gray, cmap='viridis', vmin=0, vmax=1)

            ax.axis('off')

            plt.tight_layout()
            save_path = os.path.join(vis_folder, f'contribution_cfg_{cfg_values[i]}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: contribution_cfg_{cfg_values[i]}.png")

        print(f"  All weight visualizations saved to {vis_folder}")

    def fuse(self, images_srgb, cfg_values, output_path=None, vis_folder=None):
        """
        Fuse multiple linear images into HDR
        """
        images_linear = []
        for img_srgb in images_srgb:
            img_linear = self.srgb_to_linear(img_srgb)
            images_linear.append(img_linear)

        if not images_linear or not cfg_values:
            print("  Error: No images or CFG values provided")
            return None

        print(f"  Fusing {len(images_linear)} images...")

        estimated_gains = self.estimate_exposure_gradient_based(images_linear, cfg_values)

        images_radiance = []
        for i, img_linear in enumerate(images_linear):
            img_radiance = img_linear / estimated_gains[i]
            images_radiance.append(img_radiance)

        weights = []
        for img_srgb in images_srgb:
            weight = self.compute_quality_weight(img_srgb)
            weights.append(weight)

        if self.visualize_weights and vis_folder:
            self._visualize_weights(images_srgb, images_linear, weights, cfg_values, vis_folder)

        print(f"  Building pyramids with {self.PYRAMID_LEVELS} levels...")
        hdr_final = self.laplacian_pyramid_fusion(images_radiance, weights)

        min_cfg_index = cfg_values.index(min(cfg_values))
        self.recover_saturated_regions(hdr_final, images_radiance, weights, min_cfg_index)

        if output_path:
            self.save_hdr(hdr_final, output_path)

        return hdr_final

    def laplacian_pyramid_fusion(self, images, weights):
        """Fuse using Laplacian pyramid"""
        n_images = len(images)
        h, w, c = images[0].shape

        max_levels = int(np.log2(min(h, w))) - 2
        levels = min(self.PYRAMID_LEVELS, max_levels)
        print(f"  Using {levels} pyramid levels")

        weight_pyramids = []
        for weight in weights:
            weight_gray = np.mean(weight, axis=2)
            pyramid = self.build_gaussian_pyramid(weight_gray, levels)
            weight_pyramids.append(pyramid)

        image_pyramids = []
        for img in images:
            pyramid = self.build_laplacian_pyramid(img, levels)
            image_pyramids.append(pyramid)

        blended_pyramid = []
        for level in range(levels + 1):
            numerator = np.zeros_like(image_pyramids[0][level], dtype=np.float32)
            denominator = np.zeros(image_pyramids[0][level].shape[:2], dtype=np.float32)

            for i in range(n_images):
                w = weight_pyramids[i][level]
                img = image_pyramids[i][level]

                w_expanded = w[:, :, np.newaxis]
                numerator += w_expanded * img
                denominator += w

            denominator_expanded = denominator[:, :, np.newaxis] + 1e-10
            blended = numerator / denominator_expanded
            blended_pyramid.append(blended)

        result = self.reconstruct_from_laplacian(blended_pyramid)
        return result

    def build_gaussian_pyramid(self, image, levels):
        """Build Gaussian pyramid"""
        pyramid = [image.copy()]
        current = image.copy()
        for i in range(levels):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    def build_laplacian_pyramid(self, image, levels):
        """Build Laplacian pyramid"""
        gaussian_pyramid = []
        current = image.copy()
        gaussian_pyramid.append(current)

        for i in range(levels):
            current = cv2.pyrDown(current)
            gaussian_pyramid.append(current)

        laplacian_pyramid = []
        for i in range(levels):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian = gaussian_pyramid[i] - expanded
            laplacian_pyramid.append(laplacian)

        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid

    def reconstruct_from_laplacian(self, laplacian_pyramid):
        """Reconstruct from Laplacian pyramid"""
        current = laplacian_pyramid[-1].copy()
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
            current = cv2.pyrUp(current, dstsize=size)
            current = current + laplacian_pyramid[i]
        return current

    def recover_saturated_regions(self, hdr_result, images_linear, weights, min_cfg_index):
        """Recover saturated regions"""
        weight_sum = np.zeros_like(weights[0])
        for w in weights:
            weight_sum += w

        invalid_mask = weight_sum < 0.1
        num_invalid = np.sum(invalid_mask)

        if num_invalid > 0:
            pct = num_invalid / invalid_mask.size * 100
            print(f"  Recovering saturated regions: {pct:.4f}%")
            hdr_result[invalid_mask] = images_linear[min_cfg_index][invalid_mask]

    def save_hdr(self, hdr_linear, filepath):
        """Save HDR image using OpenEXR"""
        hdr_float32 = hdr_linear.astype(np.float32)
        hdr_float32 = np.clip(hdr_float32, 0, None)

        if filepath.endswith('.hdr'):
            hdr_bgr = cv2.cvtColor(hdr_float32, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, hdr_bgr)
            print(f"  Saved HDR: {filepath}")

        elif filepath.endswith('.exr'):
            try:
                import OpenEXR
                import Imath

                h, w = hdr_float32.shape[:2]
                header = OpenEXR.Header(w, h)
                header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
                header['channels'] = {
                    'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }
                out = OpenEXR.OutputFile(filepath, header)
                R = hdr_float32[:,:,0].astype(np.float32).tobytes()
                G = hdr_float32[:,:,1].astype(np.float32).tobytes()
                B = hdr_float32[:,:,2].astype(np.float32).tobytes()
                out.writePixels({'R': R, 'G': G, 'B': B})
                out.close()
                print(f"  Saved EXR (OpenEXR): {filepath}")

            except ImportError:
                print("  Warning: OpenEXR not available, using OpenCV fallback")
                try:
                    hdr_bgr = cv2.cvtColor(hdr_float32, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, hdr_bgr)
                    print(f"  Saved EXR (OpenCV): {filepath}")
                except Exception as e:
                    print(f"  Error saving EXR: {e}")