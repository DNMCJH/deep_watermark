"""Loss functions for watermark training."""

import torch.nn as nn


class WatermarkLoss(nn.Module):
    """Combined loss: image fidelity + watermark extraction accuracy.

    loss = MSE(encoded, original) + lambda_wm * BCE(predicted_wm, target_wm)
    """

    def __init__(self, lambda_watermark=5.0):
        super().__init__()
        self.lambda_watermark = lambda_watermark
        self.image_loss_fn = nn.MSELoss()
        self.watermark_loss_fn = nn.BCELoss()

    def forward(self, encoded_image, original_image, predicted_wm, target_wm):
        """
        Returns:
            total_loss, image_loss, watermark_loss (all scalar tensors)
        """
        image_loss = self.image_loss_fn(encoded_image, original_image)
        wm_loss = self.watermark_loss_fn(predicted_wm, target_wm)
        total = image_loss + self.lambda_watermark * wm_loss
        return total, image_loss, wm_loss
