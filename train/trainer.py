"""Trainer: manages the training loop, checkpointing, and experiment tracking."""

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from eval.metrics import compute_psnr, compute_bit_accuracy
from utils.logging_utils import create_experiment_dir, save_config, save_metrics, append_experiment_log


class Trainer:
    """End-to-end training manager for the watermark pipeline."""

    def __init__(self, encoder, decoder, attack_layer, criterion,
                 train_loader, val_loader, config, device="cuda"):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.attack_layer = attack_layer.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config["learning_rate"])

        self.exp_dir = create_experiment_dir(config.get("experiment_dir", "experiments"))
        save_config(self.exp_dir, config)

    def train(self):
        """Run the full training loop."""
        num_epochs = self.config["num_epochs"]
        log_interval = self.config.get("log_interval", 50)
        save_interval = self.config.get("save_interval", 10)
        best_bit_acc = 0.0
        all_metrics = []

        for epoch in range(1, num_epochs + 1):
            train_metrics = self._train_epoch(epoch, log_interval)
            val_metrics = self._validate(epoch)

            epoch_metrics = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            all_metrics.append(epoch_metrics)
            save_metrics(self.exp_dir, all_metrics)

            print(f"Epoch {epoch}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val PSNR: {val_metrics['psnr']:.2f} | "
                  f"Val BitAcc: {val_metrics['bit_accuracy']:.4f}")

            if epoch % save_interval == 0:
                self._save_checkpoint(epoch, self.exp_dir / f"checkpoint_{epoch}.pt")

            if val_metrics["bit_accuracy"] > best_bit_acc:
                best_bit_acc = val_metrics["bit_accuracy"]
                self._save_checkpoint(epoch, self.exp_dir / "checkpoint.pt")

        append_experiment_log(
            self.exp_dir,
            self.config,
            {"psnr": val_metrics["psnr"], "bit_accuracy": val_metrics["bit_accuracy"]},
        )
        print(f"Training complete. Best bit accuracy: {best_bit_acc:.4f}")

    def _train_epoch(self, epoch, log_interval):
        """Single training epoch."""
        self.encoder.train()
        self.decoder.train()
        self.attack_layer.train()

        total_loss = 0.0
        total_img_loss = 0.0
        total_wm_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for i, (images, watermarks) in enumerate(pbar):
            images = images.to(self.device)
            watermarks = watermarks.to(self.device)

            encoded = self.encoder(images, watermarks)
            attacked = self.attack_layer(encoded)
            predicted_wm = self.decoder(attacked)

            loss, img_loss, wm_loss = self.criterion(encoded, images, predicted_wm, watermarks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_img_loss += img_loss.item()
            total_wm_loss += wm_loss.item()
            n_batches += 1

            if (i + 1) % log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {
            "loss": total_loss / n_batches,
            "image_loss": total_img_loss / n_batches,
            "watermark_loss": total_wm_loss / n_batches,
        }

    @torch.no_grad()
    def _validate(self, epoch):
        """Validation pass."""
        self.encoder.eval()
        self.decoder.eval()
        self.attack_layer.eval()

        total_psnr = 0.0
        total_bit_acc = 0.0
        n_batches = 0

        for images, watermarks in self.val_loader:
            images = images.to(self.device)
            watermarks = watermarks.to(self.device)

            encoded = self.encoder(images, watermarks)
            predicted_wm = self.decoder(encoded)

            total_psnr += compute_psnr(encoded, images)
            total_bit_acc += compute_bit_accuracy(predicted_wm, watermarks)
            n_batches += 1

        return {
            "psnr": total_psnr / max(n_batches, 1),
            "bit_accuracy": total_bit_acc / max(n_batches, 1),
        }

    def _save_checkpoint(self, epoch, path):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
