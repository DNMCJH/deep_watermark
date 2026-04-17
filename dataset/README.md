# Dataset

Place your training and evaluation images here.

## Directory Structure

```
dataset/
├── train/    # Training images
├── val/      # Validation images (optional, auto-split from train if absent)
└── test/     # Test images for evaluation
```

## Requirements

- Images can be any common format: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- No specific naming convention required
- Images will be automatically resized to 256x256 during loading
- Recommended: at least 1000 images for training

## Suggested Datasets

- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) — 800 high-quality 2K images
- [COCO](https://cocodataset.org/) — large-scale natural images
- [ImageNet](https://www.image-net.org/) — subset of any category
