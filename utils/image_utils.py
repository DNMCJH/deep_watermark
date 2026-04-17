"""Image utility functions."""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


def load_image(path, image_size=256):
    """Load an image and return a (1, 3, H, W) tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def tensor_to_image(tensor):
    """Convert a (1, 3, H, W) or (3, H, W) tensor to a PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def save_image(tensor, path):
    """Save a tensor as an image file."""
    tensor_to_image(tensor).save(path)


def visualize_comparison(original, encoded, title="Original vs Encoded"):
    """Show original and encoded images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(tensor_to_image(original))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(tensor_to_image(encoded))
    axes[1].set_title("Encoded")
    axes[1].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    return fig
