"""Dataset loader for watermark training."""

import os
import warnings
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.watermark_generator import generate_watermark


class ImageDataset(Dataset):
    """Loads images from a directory and pairs each with a random watermark."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, root_dir, image_size=256, watermark_length=32):
        self.root_dir = Path(root_dir)
        self.watermark_length = watermark_length
        self.image_paths = sorted(
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # -> [0, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except (OSError, IOError) as e:
            warnings.warn(f"Skipping corrupted image {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        img = self.transform(img)
        wm = generate_watermark(self.watermark_length)
        return img, wm
