"""Evaluate a trained watermark model on a test set."""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from models.encoder import WatermarkEncoder
from models.decoder import WatermarkDecoder
from models.attack_layer import AttackLayer
from data.dataset import ImageDataset
from eval.metrics import compute_psnr, compute_ssim, compute_bit_accuracy, compute_ber


def main():
    parser = argparse.ArgumentParser(description="Evaluate watermark model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default="dataset/test")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    encoder = WatermarkEncoder(
        watermark_length=config["watermark_length"],
        image_size=config["image_size"],
        residual_scale=config.get("residual_scale", 0.1),
    ).to(device)
    decoder = WatermarkDecoder(watermark_length=config["watermark_length"]).to(device)

    encoder.load_state_dict(ckpt["encoder_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    encoder.eval()
    decoder.eval()

    attack_layer = AttackLayer(config)
    attack_layer.eval()

    dataset = ImageDataset(args.test_dir, config["image_size"], config["watermark_length"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # No-attack evaluation
    results_clean = _evaluate(encoder, decoder, None, loader, device)
    print("=== Clean (no attack) ===")
    _print_results(results_clean)

    # With-attack evaluation
    attack_layer.train()  # enable random attacks
    results_attacked = _evaluate(encoder, decoder, attack_layer, loader, device)
    print("=== With attacks ===")
    _print_results(results_attacked)


@torch.no_grad()
def _evaluate(encoder, decoder, attack_layer, loader, device):
    total_psnr, total_ssim, total_bit_acc, total_ber = 0, 0, 0, 0
    n = 0
    for images, watermarks in loader:
        images, watermarks = images.to(device), watermarks.to(device)
        encoded = encoder(images, watermarks)
        attacked = attack_layer(encoded) if attack_layer else encoded
        predicted_wm = decoder(attacked)

        total_psnr += compute_psnr(encoded, images)
        total_ssim += compute_ssim(encoded, images)
        total_bit_acc += compute_bit_accuracy(predicted_wm, watermarks)
        total_ber += compute_ber(predicted_wm, watermarks)
        n += 1

    return {
        "psnr": total_psnr / max(n, 1),
        "ssim": total_ssim / max(n, 1),
        "bit_accuracy": total_bit_acc / max(n, 1),
        "ber": total_ber / max(n, 1),
    }


def _print_results(results):
    print(f"  PSNR:         {results['psnr']:.2f} dB")
    print(f"  SSIM:         {results['ssim']:.4f}")
    print(f"  Bit Accuracy: {results['bit_accuracy']:.4f}")
    print(f"  BER:          {results['ber']:.4f}")


if __name__ == "__main__":
    main()
