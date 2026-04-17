"""Watermark generator: produces random binary watermark vectors."""

import torch


def generate_watermark(length=32):
    """Generate a random binary watermark vector.

    Args:
        length: number of watermark bits
    Returns:
        Tensor of shape (length,) with values in {0, 1}, dtype float32
    """
    return torch.randint(0, 2, (length,)).float()
