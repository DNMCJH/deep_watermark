"""Evaluation metrics for watermark quality and extraction accuracy."""

import torch
import numpy as np
from skimage.metrics import structural_similarity


def compute_psnr(encoded, original):
    """Compute PSNR between encoded and original images.

    Args:
        encoded: (B, 3, H, W) tensor in [0, 1]
        original: (B, 3, H, W) tensor in [0, 1]
    Returns:
        Average PSNR in dB (float)
    """
    mse = torch.mean((encoded - original) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(encoded, original):
    """Compute average SSIM between encoded and original images.

    Args:
        encoded: (B, 3, H, W) tensor in [0, 1]
        original: (B, 3, H, W) tensor in [0, 1]
    Returns:
        Average SSIM (float)
    """
    enc_np = encoded.detach().cpu().numpy().transpose(0, 2, 3, 1)
    ori_np = original.detach().cpu().numpy().transpose(0, 2, 3, 1)

    ssim_vals = []
    for e, o in zip(enc_np, ori_np):
        ssim_vals.append(structural_similarity(e, o, channel_axis=2, data_range=1.0))
    return float(np.mean(ssim_vals))


def compute_bit_accuracy(predicted_wm, target_wm):
    """Compute bit-level accuracy.

    Args:
        predicted_wm: (B, L) tensor in [0, 1] (sigmoid output)
        target_wm: (B, L) tensor in {0, 1}
    Returns:
        Accuracy in [0, 1] (float)
    """
    pred_bits = (predicted_wm > 0.5).float()
    correct = (pred_bits == target_wm).float()
    return correct.mean().item()


def compute_ber(predicted_wm, target_wm):
    """Compute Bit Error Rate.

    Args:
        predicted_wm: (B, L) tensor in [0, 1]
        target_wm: (B, L) tensor in {0, 1}
    Returns:
        BER in [0, 1] (float)
    """
    return 1.0 - compute_bit_accuracy(predicted_wm, target_wm)
