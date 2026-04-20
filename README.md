# Deep Watermark — AIGC Image Copyright Protection

[中文文档](README_CN.md)

A deep learning-based watermarking system for embedding and extracting robust watermarks in images.

## Overview

This project implements an end-to-end trainable watermarking pipeline:

```
Image + Watermark → Encoder → Watermarked Image → Attack Layer → Decoder → Recovered Watermark
```

**Current stage**: Baseline system with simple attacks (noise, blur, JPEG, resize, crop).

**Target metrics**: PSNR > 35 dB, Bit Accuracy > 80%

## Setup

### 1. Virtual Environment

```bash
python -m venv watermark_env
# Windows
watermark_env\Scripts\activate
# Linux/Mac
source watermark_env/bin/activate
```

### 2. Install Dependencies

**CPU only** (no GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**CUDA 11.8** (with NVIDIA GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> Note: The code auto-detects GPU availability. CPU and CUDA environments use the same codebase, no changes needed.

### 3. Prepare Dataset

Download DIV2K automatically (supports resume on interruption):
```bash
python scripts/download_div2k.py
```

The script resumes from where it left off if interrupted — just re-run it. It retries up to 10 times with exponential backoff, verifies zip integrity before extraction, and skips already-extracted images.

Or manually place training images in `dataset/train/`. See [dataset/README.md](dataset/README.md) for details.

## Usage

### Training

```bash
python -m train.train --config configs/train.yaml
```

### Evaluation

```bash
python -m eval.evaluate --checkpoint experiments/<exp_id>/checkpoint.pt --test_dir dataset/test
```

### Demo

```bash
streamlit run demo/streamlit_app.py
```

## Project Structure

```
├── configs/train.yaml          # Training configuration
├── data/
│   ├── dataset.py              # Image dataset loader
│   └── watermark_generator.py  # Random watermark generation
├── models/
│   ├── encoder.py              # Watermark encoder (residual learning)
│   ├── decoder.py              # Watermark decoder (CNN + FC)
│   └── attack_layer.py         # Differentiable attack simulation
├── train/
│   ├── train.py                # Training entry point
│   ├── trainer.py              # Training loop & checkpointing
│   └── loss.py                 # Combined image + watermark loss
├── eval/
│   ├── metrics.py              # PSNR, SSIM, Bit Accuracy, BER
│   └── evaluate.py             # Evaluation script
├── utils/
│   ├── image_utils.py          # Image I/O and visualization
│   └── logging_utils.py        # Experiment tracking utilities
├── demo/streamlit_app.py       # Interactive demo
├── experiments/                # Auto-generated experiment logs
└── dataset/                    # Training/test images
```

## Git Workflow

```bash
git init
git remote add origin <your-repo-url>
git checkout -b dev
git push -u origin dev
```

Branch structure:
- `main` — stable releases
- `dev` — active development
- `feature/*` — individual features

## License

See [LICENSE](LICENSE).
