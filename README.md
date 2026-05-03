# Stable Diffusion Inpainting — LoRA Fine-Tuning Demo

A self-contained demo that fine-tunes Stable Diffusion's inpainting model using LoRA on a small dataset, then compares before/after results with quantitative metrics.

Designed to run comfortably on a MacBook with 24GB RAM using the MPS (Metal) backend.

## What It Does

1. **Downloads** a small public dataset from HuggingFace (`diffusers/dog-example`)
2. **Generates** random masks for inpainting training pairs
3. **Fine-tunes** the SD inpainting UNet with LoRA (only ~0.1% of parameters)
4. **Runs inference** with both the base and fine-tuned models
5. **Evaluates** results using PSNR, SSIM, and LPIPS metrics
6. **Generates** comparison images, plots, and a markdown report

## Quick Start

```bash
# 1. Create the conda environment
bash setup.sh

# 2. Activate it
conda activate sd-inpaint-finetune

# 3. Run the full pipeline
python run_demo.py
```

## Output Structure

```
outputs/
├── samples/              # Training data samples
├── lora_weights/         # Saved LoRA adapter weights
├── results/              # Side-by-side comparison images
│   ├── comparison_0.png  # Strip: original | mask | masked | base | finetuned
│   ├── base_output_0.png
│   └── finetuned_output_0.png
└── evaluation/
    ├── metrics.json      # Raw metric values
    ├── report.md         # Formatted results report
    ├── training_loss.png # Loss curve plot
    ├── metrics_comparison.png  # Bar chart of metrics
    └── comparison_grid.png     # Labeled comparison grid
```

## Configuration

Edit `config.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 256 | Resolution (lower = faster, less RAM) |
| `lora_rank` | 4 | LoRA rank (higher = more capacity) |
| `max_train_steps` | 50 | Total training steps |
| `num_train_samples` | 5 | Training images |
| `num_eval_samples` | 2 | Evaluation images |
| `num_inference_steps` | 30 | Diffusion steps at inference |

## Requirements

- macOS with Apple Silicon (M1/M2/M3) — uses MPS backend
- 24GB RAM (runs fine with these default settings)
- ~5GB disk space for model weights
- conda or miniconda

## How It Works

**LoRA (Low-Rank Adaptation)** adds small trainable matrices to the attention layers of the UNet, keeping 99.9% of parameters frozen. This means:
- Training is fast (minutes, not hours)
- Memory usage stays low
- The adapter weights are tiny (~2MB vs 3.4GB for the full model)

The training loop:
1. Encode images to latent space using the VAE
2. Add noise at random timesteps
3. Concatenate noisy latents + mask + masked latents (9-channel input)
4. Predict the noise with the UNet
5. Backpropagate only through LoRA parameters

## Metrics Explained

- **PSNR** (Peak Signal-to-Noise Ratio): Pixel-level similarity. Higher = better.
- **SSIM** (Structural Similarity): Structural preservation. Higher = better.
- **LPIPS** (Learned Perceptual Similarity): Perceptual distance via neural net. Lower = better.
