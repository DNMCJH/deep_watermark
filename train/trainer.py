"""Trainer: manages the training loop, checkpointing, and experiment tracking."""

import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from eval.metrics import compute_psnr, compute_bit_accuracy
from utils.logging_utils import create_experiment_dir, save_config, save_metrics, append_experiment_log


class Trainer:
    """End-to-end training manager for the watermark pipeline."""

    def __init__(self, encoder, decoder, attack_layer, criterion,
                 train_loader, val_loader, config, device="cuda",
                 resume_path=None):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.attack_layer = attack_layer.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_amp = device == "cuda"

        params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config["learning_rate"])
        self.grad_clip_norm = config.get("grad_clip_norm", 1.0)
        self.scaler = GradScaler(enabled=self.use_amp)

        # LR scheduler
        num_epochs = config["num_epochs"]
        self.warmup_epochs = config.get("warmup_epochs", 0)
        scheduler_type = config.get("scheduler", "cosine")

        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(num_epochs - self.warmup_epochs, 1)
            )
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=10, factor=0.5
            )
        else:
            self.scheduler = None

        self.start_epoch = 1
        self.best_bit_acc = 0.0
        self.all_metrics = []

        self.exp_dir = create_experiment_dir(config.get("experiment_dir", "experiments"))
        save_config(self.exp_dir, config)

        if resume_path:
            self._load_checkpoint(resume_path)

    def _load_checkpoint(self, path):
        """Resume training from a checkpoint."""
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.decoder.load_state_dict(ckpt["decoder_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_bit_acc = ckpt.get("best_bit_acc", 0.0)
        print(f"  Resumed at epoch {self.start_epoch}, best BitAcc={self.best_bit_acc:.4f}")

    def train(self):
        """Run the full training loop."""
        num_epochs = self.config["num_epochs"]
        log_interval = self.config.get("log_interval", 50)
        save_interval = self.config.get("save_interval", 10)

        for epoch in range(self.start_epoch, num_epochs + 1):
            # Warmup: linear LR ramp
            if epoch <= self.warmup_epochs:
                warmup_lr = self.config["learning_rate"] * epoch / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            train_metrics = self._train_epoch(epoch, log_interval)
            val_metrics = self._validate(epoch)

            # Step scheduler after warmup
            if epoch > self.warmup_epochs and self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["bit_accuracy"])
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            epoch_metrics = {
                "epoch": epoch,
                "lr": lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.all_metrics.append(epoch_metrics)
            save_metrics(self.exp_dir, self.all_metrics)

            print(f"Epoch {epoch}/{num_epochs} | "
                  f"LR: {lr:.6f} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val PSNR: {val_metrics['psnr']:.2f} | "
                  f"Val BitAcc: {val_metrics['bit_accuracy']:.4f}")

            if epoch % save_interval == 0:
                self._save_checkpoint(epoch, self.exp_dir / f"checkpoint_{epoch}.pt")

            if val_metrics["bit_accuracy"] > self.best_bit_acc:
                self.best_bit_acc = val_metrics["bit_accuracy"]
                self._save_checkpoint(epoch, self.exp_dir / "checkpoint.pt")

        append_experiment_log(
            self.exp_dir,
            self.config,
            {"psnr": val_metrics["psnr"], "bit_accuracy": val_metrics["bit_accuracy"]},
        )
        print(f"Training complete. Best bit accuracy: {self.best_bit_acc:.4f}")

    def _train_epoch(self, epoch, log_interval):
        """Single training epoch with mixed precision and gradient clipping."""
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

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                encoded = self.encoder(images, watermarks)
                attacked = self.attack_layer(encoded)
                predicted_wm = self.decoder(attacked)
                loss, img_loss, wm_loss = self.criterion(encoded, images, predicted_wm, watermarks)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                self.grad_clip_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
        """Save model checkpoint with full training state for resume."""
        state = {
            "epoch": epoch,
            "best_bit_acc": self.best_bit_acc,
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, path)
