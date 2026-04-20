"""Attack Layer: applies random differentiable distortions during training."""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from kornia.enhance import jpeg_codec_differentiable


def gaussian_noise(image, std=0.02):
    """Add Gaussian noise."""
    return image + torch.randn_like(image) * std


def gaussian_blur(image, kernel_size=5):
    """Apply Gaussian blur."""
    blur = K.RandomGaussianBlur(
        kernel_size=(kernel_size, kernel_size),
        sigma=(0.1, 2.0),
        p=1.0,
    )
    return blur(image)


def resize_attack(image, ratio=0.7):
    """Downscale then upscale back to original size."""
    _, _, h, w = image.shape
    small_h, small_w = int(h * ratio), int(w * ratio)
    down = F.interpolate(image, size=(small_h, small_w), mode="bilinear", align_corners=False)
    return F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


def crop_attack(image, ratio=0.8):
    """Random crop and resize back to original size."""
    _, _, h, w = image.shape
    crop_h, crop_w = int(h * ratio), int(w * ratio)
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    cropped = image[:, :, top:top + crop_h, left:left + crop_w]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


def jpeg_attack(image, quality=50):
    """Differentiable JPEG approximation using kornia."""
    q = torch.tensor([quality], dtype=image.dtype, device=image.device)
    return jpeg_codec_differentiable(image, jpeg_quality=q)


class AttackLayer(nn.Module):
    """Randomly applies one distortion during training. Identity during eval."""

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}
        self.noise_std = cfg.get("noise_std", 0.02)
        self.jpeg_quality = cfg.get("jpeg_quality", 50)
        self.blur_kernel_size = cfg.get("blur_kernel_size", 5)
        self.resize_ratio = cfg.get("resize_ratio", 0.7)
        self.crop_ratio = cfg.get("crop_ratio", 0.8)
        self.attack_prob = cfg.get("attack_prob", 0.8)

        self.attacks = {
            "gaussian_noise": lambda x: gaussian_noise(x, self.noise_std),
            "jpeg_compression": lambda x: jpeg_attack(x, self.jpeg_quality),
            "gaussian_blur": lambda x: gaussian_blur(x, self.blur_kernel_size),
            "resize": lambda x: resize_attack(x, self.resize_ratio),
            "crop": lambda x: crop_attack(x, self.crop_ratio),
        }

        enabled = cfg.get("attacks", list(self.attacks.keys()))
        self.enabled_attacks = [a for a in enabled if a in self.attacks]
        if not self.enabled_attacks:
            raise ValueError("No valid attacks enabled. Check 'attacks' in config.")

    def forward(self, image):
        """
        Args:
            image: (B, 3, H, W) in [0, 1]
        Returns:
            attacked image, same shape
        """
        if not self.training or random.random() > self.attack_prob:
            return image

        attack_name = random.choice(self.enabled_attacks)
        attacked = self.attacks[attack_name](image)
        return torch.clamp(attacked, 0.0, 1.0)
