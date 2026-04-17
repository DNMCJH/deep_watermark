"""Experiment tracking utilities."""

import json
from datetime import datetime
from pathlib import Path

import yaml


def create_experiment_dir(base_dir="experiments"):
    """Create a timestamped experiment directory.

    Returns:
        Path to the new experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(exp_dir, config):
    """Save training config to experiment directory."""
    path = Path(exp_dir) / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_metrics(exp_dir, metrics):
    """Save metrics list to experiment directory."""
    path = Path(exp_dir) / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def append_experiment_log(exp_dir, config, final_metrics):
    """Append a summary line to the global experiment log."""
    log_path = Path(exp_dir).parent / "experiment_log.md"

    if not log_path.exists():
        log_path.write_text(
            "# Experiment Log\n\n"
            "| Experiment ID | PSNR | Bit Accuracy | Lambda | LR | Notes |\n"
            "|---|---|---|---|---|---|\n"
        )

    exp_id = Path(exp_dir).name
    psnr = final_metrics.get("psnr", 0)
    bit_acc = final_metrics.get("bit_accuracy", 0)
    lam = config.get("lambda_watermark", "")
    lr = config.get("learning_rate", "")

    line = f"| {exp_id} | {psnr:.2f} | {bit_acc:.4f} | {lam} | {lr} | |\n"

    with open(log_path, "a") as f:
        f.write(line)
