"""Training entry point."""

import argparse
import random

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from models.encoder import WatermarkEncoder
from models.decoder import WatermarkDecoder
from models.attack_layer import AttackLayer
from train.loss import WatermarkLoss
from train.trainer import Trainer
from data.dataset import ImageDataset


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train deep watermark baseline")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    config = load_config(args.config)

    seed = config.get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device} | Seed: {seed}")

    # Dataset
    dataset = ImageDataset(
        root_dir=config["train_dir"],
        image_size=config["image_size"],
        watermark_length=config["watermark_length"],
    )
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"],
        shuffle=True, num_workers=config.get("num_workers", 4),
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"],
        shuffle=False, num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    # Models
    encoder = WatermarkEncoder(
        watermark_length=config["watermark_length"],
        image_size=config["image_size"],
        residual_scale=config.get("residual_scale", 0.1),
    )
    decoder = WatermarkDecoder(watermark_length=config["watermark_length"])
    attack_layer = AttackLayer(config)
    criterion = WatermarkLoss(lambda_watermark=config.get("lambda_watermark", 5.0))

    # Train
    trainer = Trainer(
        encoder, decoder, attack_layer, criterion,
        train_loader, val_loader, config, device,
        resume_path=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
