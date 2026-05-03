"""
Evaluate inpainting quality: compare base vs fine-tuned model outputs.
Metrics: PSNR, SSIM, LPIPS (perceptual similarity).
"""

import json

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from config import Config


def compute_metrics(original: Image.Image, generated: Image.Image) -> dict:
    """Compute image quality metrics between original and generated images."""
    # Resize to match if needed
    if original.size != generated.size:
        generated = generated.resize(original.size, Image.LANCZOS)

    orig_np = np.array(original).astype(np.float64)
    gen_np = np.array(generated).astype(np.float64)

    # PSNR (higher is better)
    psnr_val = psnr(orig_np, gen_np, data_range=255.0)

    # SSIM (higher is better)
    ssim_val = ssim(
        orig_np, gen_np,
        channel_axis=2,
        data_range=255.0,
    )

    return {
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
    }


def compute_lpips(original: Image.Image, generated: Image.Image, loss_fn) -> float:
    """Compute LPIPS perceptual distance (lower is better)."""
    def to_tensor(img):
        arr = np.array(img.resize((256, 256))).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # scale to [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        t_orig = to_tensor(original)
        t_gen = to_tensor(generated)
        dist = loss_fn(t_orig, t_gen)
    return float(dist.item())


def evaluate(
    eval_data: list,
    base_results: list,
    finetuned_results: list,
    cfg: Config,
) -> dict:
    """Run full evaluation comparing base vs fine-tuned outputs."""
    print("\nEvaluating inpainting quality...")
    print("=" * 60)

    # Load LPIPS model (uses VGG by default)
    loss_fn = lpips.LPIPS(net="squeeze")  # squeeze is smallest

    base_metrics = []
    ft_metrics = []

    for i in range(len(eval_data)):
        original = eval_data[i]["image"]
        base_out = base_results[i]
        ft_out = finetuned_results[i]

        # Compute metrics for base model
        bm = compute_metrics(original, base_out)
        bm["lpips"] = compute_lpips(original, base_out, loss_fn)
        base_metrics.append(bm)

        # Compute metrics for fine-tuned model
        fm = compute_metrics(original, ft_out)
        fm["lpips"] = compute_lpips(original, ft_out, loss_fn)
        ft_metrics.append(fm)

        print(f"\nSample {i + 1}:")
        print(f"  Base     -> PSNR: {bm['psnr']:.2f} | SSIM: {bm['ssim']:.4f} | LPIPS: {bm['lpips']:.4f}")
        print(f"  FineTune -> PSNR: {fm['psnr']:.2f} | SSIM: {fm['ssim']:.4f} | LPIPS: {fm['lpips']:.4f}")

    # Averages
    avg_base = {
        k: np.mean([m[k] for m in base_metrics]) for k in ["psnr", "ssim", "lpips"]
    }
    avg_ft = {
        k: np.mean([m[k] for m in ft_metrics]) for k in ["psnr", "ssim", "lpips"]
    }

    print("\n" + "=" * 60)
    print("AVERAGE RESULTS")
    print("=" * 60)
    print(f"  {'Metric':<10} {'Base':>10} {'FineTuned':>12} {'Better?':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    for metric in ["psnr", "ssim", "lpips"]:
        b = avg_base[metric]
        f = avg_ft[metric]
        # For PSNR and SSIM, higher is better. For LPIPS, lower is better.
        if metric == "lpips":
            better = "FineTuned" if f < b else "Base"
        else:
            better = "FineTuned" if f > b else "Base"
        print(f"  {metric.upper():<10} {b:>10.4f} {f:>12.4f} {better:>10}")

    results = {
        "per_sample": {
            "base": base_metrics,
            "finetuned": ft_metrics,
        },
        "averages": {
            "base": {k: float(v) for k, v in avg_base.items()},
            "finetuned": {k: float(v) for k, v in avg_ft.items()},
        },
    }

    # Save results
    results_path = cfg.eval_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {results_path}")

    return results


def generate_report(results: dict, cfg: Config):
    """Generate a markdown report with evaluation results."""
    avg_b = results["averages"]["base"]
    avg_f = results["averages"]["finetuned"]

    report = f"""# SD Inpainting Fine-Tuning Results

## Configuration
- Model: `{cfg.model_id}`
- LoRA rank: {cfg.lora_rank}
- Training steps: {cfg.max_train_steps}
- Image size: {cfg.image_size}x{cfg.image_size}
- Train samples: {cfg.num_train_samples}
- Eval samples: {cfg.num_eval_samples}
- Prompt: "{cfg.inpaint_prompt}"

## Average Metrics

| Metric | Base Model | Fine-Tuned | Direction |
|--------|-----------|------------|-----------|
| PSNR   | {avg_b['psnr']:.4f} | {avg_f['psnr']:.4f} | Higher = Better |
| SSIM   | {avg_b['ssim']:.4f} | {avg_f['ssim']:.4f} | Higher = Better |
| LPIPS  | {avg_b['lpips']:.4f} | {avg_f['lpips']:.4f} | Lower = Better |

## Per-Sample Results

"""
    for i, (bm, fm) in enumerate(zip(
        results["per_sample"]["base"],
        results["per_sample"]["finetuned"],
    )):
        report += f"""### Sample {i + 1}
| Metric | Base | Fine-Tuned |
|--------|------|------------|
| PSNR   | {bm['psnr']:.4f} | {fm['psnr']:.4f} |
| SSIM   | {bm['ssim']:.4f} | {fm['ssim']:.4f} |
| LPIPS  | {bm['lpips']:.4f} | {fm['lpips']:.4f} |

![Comparison](results/comparison_{i}.png)

"""

    report += """## How to Read Results
- **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level similarity. Higher = more similar to original.
- **SSIM** (Structural Similarity Index): Measures structural similarity. Higher = better preserved structure.
- **LPIPS** (Learned Perceptual Image Patch Similarity): Measures perceptual similarity using a neural network. Lower = more perceptually similar.

## Notes
- This is a small-scale demo with limited training data and steps.
- With more data and training, fine-tuned results would improve significantly.
- The fine-tuned model learns to better match the style/content of the training domain.
"""

    report_path = cfg.eval_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")
