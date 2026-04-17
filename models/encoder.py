"""Watermark Encoder: embeds a binary watermark into an image via learned residual."""

import torch
import torch.nn as nn


class WatermarkEncoder(nn.Module):
    """Embeds watermark bits into an image using residual learning.

    Pipeline: watermark -> linear -> spatial expand -> concat with image -> CNN -> residual
    Output: image + scale * residual, clamped to [0, 1]
    """

    def __init__(self, watermark_length=32, image_size=256, residual_scale=0.1):
        super().__init__()
        self.image_size = image_size
        self.residual_scale = residual_scale

        self.watermark_fc = nn.Linear(watermark_length, image_size * image_size)

        # input: 4 channels (3 RGB + 1 watermark feature map)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, image, watermark):
        """
        Args:
            image: (B, 3, H, W) in [0, 1]
            watermark: (B, watermark_length) binary
        Returns:
            encoded_image: (B, 3, H, W) in [0, 1]
        """
        wm_expanded = self.watermark_fc(watermark)
        wm_map = wm_expanded.view(-1, 1, self.image_size, self.image_size)

        x = torch.cat([image, wm_map], dim=1)
        residual = self.conv(x)

        encoded = image + self.residual_scale * residual
        return torch.clamp(encoded, 0.0, 1.0)
