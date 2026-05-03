"""
Generate visualization plots for the fine-tuning demo.
"""

import json

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import Config


def plot_training_loss(losses: list, cfg: Config):
    """Plot training loss curve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, "b-", linewidth=1.5, alpha=0.7)

    # Add smoothed line
    if len(losses) > 5:
        window = min(5, len(losses) // 3)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        offset = window // 2
        ax.plot(
            range(1 + offset, len(smoothed) + 1 + offset),
            smoothed, "r-", linewidth=2, label="Smoothed"
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("LoRA Fine-Tuning Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = cfg.eval_dir / "training_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss plot saved to {path}")


def plot_metrics_comparison(cfg: Config):
    """Plot bar chart comparing base vs fine-tuned metrics."""
    metrics_path = cfg.eval_dir / "metrics.json"
    if not metrics_path.exists():
        print("  No metrics file found, skipping metrics plot")
        return

    with open(metrics_path) as f:
        results = json.load(f)

    avg_b = results["averages"]["base"]
    avg_f = results["averages"]["finetuned"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, metric, title, higher_better in zip(
        axes,
        ["psnr", "ssim", "lpips"],
        ["PSNR (dB)", "SSIM", "LPIPS"],
        [True, True, False],
    ):
        vals = [avg_b[metric], avg_f[metric]]
        colors = ["#4A90D9", "#E8744F"]
        bars = ax.bar(["Base", "Fine-Tuned"], vals, color=colors, width=0.5)

        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=10
            )

        ax.set_title(title)
        direction = "↑ Higher = Better" if higher_better else "↓ Lower = Better"
        ax.set_xlabel(direction, fontsize=8, color="gray")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Base vs Fine-Tuned Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = cfg.eval_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Metrics comparison saved to {path}")


def create_comparison_grid(cfg: Config):
    """Create a labeled comparison grid from saved results."""
    results_dir = cfg.results_dir
    comparisons = sorted(results_dir.glob("comparison_*.png"))

    if not comparisons:
        print("  No comparison images found")
        return

    fig, axes = plt.subplots(
        len(comparisons), 1,
        figsize=(15, 4 * len(comparisons))
    )
    if len(comparisons) == 1:
        axes = [axes]

    labels = ["Original", "Mask", "Masked Input", "Base Output", "Fine-Tuned"]

    for ax, comp_path in zip(axes, comparisons):
        img = Image.open(comp_path)
        ax.imshow(np.array(img))
        ax.axis("off")

        # Add column labels
        w = img.size[0] // 5
        for j, label in enumerate(labels):
            ax.text(
                j * w + w // 2, -10, label,
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

    fig.suptitle(
        "Inpainting Results: Base vs Fine-Tuned",
        fontsize=16, fontweight="bold", y=1.02
    )
    fig.tight_layout()

    path = cfg.eval_dir / "comparison_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison grid saved to {path}")


if __name__ == "__main__":
    cfg = Config()
    plot_metrics_comparison(cfg)
    create_comparison_grid(cfg)
