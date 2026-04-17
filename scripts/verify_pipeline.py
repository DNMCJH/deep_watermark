"""可视化验证：用真实图片跑完整 pipeline，生成对比图。"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from models.encoder import WatermarkEncoder
from models.decoder import WatermarkDecoder
from models.attack_layer import AttackLayer, gaussian_noise, gaussian_blur, resize_attack, crop_attack
from eval.metrics import compute_psnr, compute_bit_accuracy, compute_ssim
from utils.image_utils import load_image, tensor_to_image


def main():
    device = "cpu"

    # 1. 加载模型（随机初始化，仅验证结构）
    encoder = WatermarkEncoder(watermark_length=32, image_size=256).to(device)
    decoder = WatermarkDecoder(watermark_length=32).to(device)
    print("[OK] Encoder and Decoder initialized")
    print(f"     Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"     Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    # 2. 加载真实图片
    train_dir = "dataset/train"
    images = sorted([f for f in os.listdir(train_dir) if f.endswith(".png")])[:3]
    if not images:
        print("[ERROR] No images found in dataset/train/")
        return

    # 3. 定义水印
    watermark = torch.tensor([1,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1,
                              0,1,1,0,1,0,0,1,1,0,1,1,0,0,1,0], dtype=torch.float32)
    wm_str = "".join(str(int(b)) for b in watermark)
    print(f"\n[INFO] Watermark: {wm_str}")

    # 4. 对每张图跑完整 pipeline 并可视化
    os.makedirs("assets/verify", exist_ok=True)

    for img_name in images:
        img_path = os.path.join(train_dir, img_name)
        image = load_image(img_path, image_size=256).to(device)
        wm = watermark.unsqueeze(0).to(device)

        with torch.no_grad():
            encoded = encoder(image, wm)
            noised = gaussian_noise(encoded, std=0.02)
            blurred = gaussian_blur(encoded, kernel_size=5)
            resized = resize_attack(encoded, ratio=0.7)
            cropped = crop_attack(encoded, ratio=0.8)

            pred_clean = decoder(encoded)
            pred_noised = decoder(noised)
            pred_blurred = decoder(blurred)

        psnr = compute_psnr(encoded, image)
        ssim = compute_ssim(encoded, image)
        bit_acc_clean = compute_bit_accuracy(pred_clean, wm)
        bit_acc_noised = compute_bit_accuracy(pred_noised, wm)
        bit_acc_blurred = compute_bit_accuracy(pred_blurred, wm)

        # 生成对比图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Pipeline Verification - {img_name}", fontsize=14)

        panels = [
            (image, "Original"),
            (encoded, f"Watermarked\nPSNR={psnr:.1f}dB SSIM={ssim:.3f}"),
            (encoded - image, "Residual (x10)"),
            (noised, f"+ Noise\nBitAcc={bit_acc_noised:.2%}"),
            (blurred, f"+ Blur\nBitAcc={bit_acc_blurred:.2%}"),
            (cropped, f"+ Crop\nBitAcc(clean)={bit_acc_clean:.2%}"),
        ]

        for ax, (tensor, title) in zip(axes.flat, panels):
            if "Residual" in title:
                diff = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 10 + 0.5).clip(0, 1)
                ax.imshow(diff)
            else:
                ax.imshow(tensor_to_image(tensor))
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        plt.tight_layout()
        save_path = f"assets/verify/{img_name.replace('.png', '_verify.png')}"
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[OK] {img_name}")
        print(f"     PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")
        print(f"     BitAcc - clean: {bit_acc_clean:.2%} | noised: {bit_acc_noised:.2%} | blurred: {bit_acc_blurred:.2%}")
        print(f"     Saved: {save_path}")

    print(f"\n=== Done! Check assets/verify/ for visual results ===")


if __name__ == "__main__":
    main()
