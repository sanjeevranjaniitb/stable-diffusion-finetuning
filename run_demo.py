#!/usr/bin/env python3
"""
Full pipeline: download data → train LoRA → inference → evaluate → visualize.
Run this after activating the conda environment.

Usage:
    conda activate sd-inpaint-finetune
    python run_demo.py
"""

import gc
import sys
import time

import torch


def main():
    total_start = time.time()

    print("=" * 60)
    print("  Stable Diffusion Inpainting - LoRA Fine-Tuning Demo")
    print("=" * 60)

    # Check environment
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device == "cpu":
        print("WARNING: MPS not available, running on CPU (will be slow)")
    print()

    # ── Step 1: Prepare Data ──────────────────────────────────
    print("=" * 60)
    print("STEP 1: Preparing Dataset")
    print("=" * 60)
    from config import Config
    from data_prep import prepare_dataset

    cfg = Config()
    train_data, eval_data = prepare_dataset(cfg)

    # Save sample images
    sample_dir = cfg.output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    for i, entry in enumerate(train_data[:3]):
        entry["image"].save(sample_dir / f"train_{i}_original.png")
        entry["mask"].save(sample_dir / f"train_{i}_mask.png")
        entry["masked_image"].save(sample_dir / f"train_{i}_masked.png")
    print(f"Sample images saved to {sample_dir}/\n")

    # ── Step 2: Train LoRA ────────────────────────────────────
    print("=" * 60)
    print("STEP 2: Fine-Tuning with LoRA")
    print("=" * 60)
    from train import train

    losses, _ = train(cfg)
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    print()

    # ── Step 3: Run Inference ─────────────────────────────────
    print("=" * 60)
    print("STEP 3: Running Inference (Base vs Fine-Tuned)")
    print("=" * 60)
    from inference import run_base_inference, run_finetuned_inference, save_comparison

    print("\n--- Base Model ---")
    base_results = run_base_inference(cfg, eval_data)
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    print("\n--- Fine-Tuned Model ---")
    ft_results = run_finetuned_inference(cfg, eval_data)
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    save_comparison(eval_data, base_results, ft_results, cfg)
    print()

    # ── Step 4: Evaluate ──────────────────────────────────────
    print("=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)
    from evaluate import evaluate, generate_report

    results = evaluate(eval_data, base_results, ft_results, cfg)
    generate_report(results, cfg)
    print()

    # ── Step 5: Visualize ─────────────────────────────────────
    print("=" * 60)
    print("STEP 5: Generating Visualizations")
    print("=" * 60)
    from visualize import plot_training_loss, plot_metrics_comparison, create_comparison_grid

    plot_training_loss(losses, cfg)
    plot_metrics_comparison(cfg)
    create_comparison_grid(cfg)
    print()

    # ── Done ──────────────────────────────────────────────────
    total_time = time.time() - total_start
    print("=" * 60)
    print("  DEMO COMPLETE!")
    print("=" * 60)
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"\n  Output files:")
    print(f"    Training loss plot:   {cfg.eval_dir}/training_loss.png")
    print(f"    Metrics comparison:   {cfg.eval_dir}/metrics_comparison.png")
    print(f"    Comparison grid:      {cfg.eval_dir}/comparison_grid.png")
    print(f"    Detailed report:      {cfg.eval_dir}/report.md")
    print(f"    Metrics JSON:         {cfg.eval_dir}/metrics.json")
    print(f"    LoRA weights:         {cfg.lora_weights_dir}/")
    print(f"    Side-by-side images:  {cfg.results_dir}/")
    print()


if __name__ == "__main__":
    main()
