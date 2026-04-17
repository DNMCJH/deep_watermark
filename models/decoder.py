"""Watermark Decoder: extracts watermark bits from a (possibly attacked) image."""

import torch.nn as nn


class WatermarkDecoder(nn.Module):
    """Extracts watermark bits from an image.

    Architecture: 3x stride-2 Conv blocks -> Global Average Pooling -> FC -> sigmoid
    """

    def __init__(self, watermark_length=32):
        super().__init__()

        self.features = nn.Sequential(
            # (B, 3, 256, 256) -> (B, 64, 128, 128)
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # (B, 64, 128, 128) -> (B, 128, 64, 64)
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # (B, 128, 64, 64) -> (B, 256, 32, 32)
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(256, watermark_length),
            nn.Sigmoid(),
        )

    def forward(self, image):
        """
        Args:
            image: (B, 3, H, W) in [0, 1]
        Returns:
            watermark: (B, watermark_length) in [0, 1]
        """
        x = self.features(image)
        x = self.pool(x).flatten(1)
        return self.fc(x)
