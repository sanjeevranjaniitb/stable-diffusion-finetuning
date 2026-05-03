"""
Run inpainting inference with both the base model and the fine-tuned LoRA model.
Generates side-by-side comparison images.
"""

import gc

import torch
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel
from PIL import Image

from config import Config
from data_prep import prepare_dataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_base_inference(cfg: Config, eval_data: list) -> list:
    """Run inpainting with the base (non-fine-tuned) model."""
    device = get_device()
    print("Loading base inpainting pipeline...")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    # Reduce memory usage
    pipe.enable_attention_slicing()

    results = []
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)

    for i, entry in enumerate(eval_data):
        print(f"  Base inference on sample {i + 1}/{len(eval_data)}...")
        output = pipe(
            prompt=cfg.inpaint_prompt,
            image=entry["image"],
            mask_image=entry["mask"],
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generator,
            height=cfg.image_size,
            width=cfg.image_size,
        ).images[0]
        results.append(output)

    # Cleanup
    del pipe
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return results


def run_finetuned_inference(cfg: Config, eval_data: list) -> list:
    """Run inpainting with the LoRA fine-tuned model."""
    device = get_device()
    print("Loading fine-tuned inpainting pipeline...")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    )

    # Load LoRA weights into UNet
    pipe.unet = PeftModel.from_pretrained(
        pipe.unet, str(cfg.lora_weights_dir)
    )

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    results = []
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)

    for i, entry in enumerate(eval_data):
        print(f"  Fine-tuned inference on sample {i + 1}/{len(eval_data)}...")
        output = pipe(
            prompt=cfg.inpaint_prompt,
            image=entry["image"],
            mask_image=entry["mask"],
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generator,
            height=cfg.image_size,
            width=cfg.image_size,
        ).images[0]
        results.append(output)

    # Cleanup
    del pipe
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return results


def save_comparison(
    eval_data: list,
    base_results: list,
    finetuned_results: list,
    cfg: Config,
):
    """Save side-by-side comparison images."""
    for i in range(len(eval_data)):
        original = eval_data[i]["image"]
        mask = eval_data[i]["mask"]
        masked = eval_data[i]["masked_image"]
        base_out = base_results[i]
        ft_out = finetuned_results[i]

        # Create comparison strip: original | mask | masked | base | finetuned
        w, h = original.size
        strip = Image.new("RGB", (w * 5, h))
        strip.paste(original, (0, 0))
        strip.paste(mask.convert("RGB"), (w, 0))
        strip.paste(masked, (w * 2, 0))
        strip.paste(base_out.resize((w, h)), (w * 3, 0))
        strip.paste(ft_out.resize((w, h)), (w * 4, 0))

        path = cfg.results_dir / f"comparison_{i}.png"
        strip.save(path)
        print(f"  Saved comparison to {path}")

        # Also save individual outputs
        base_out.save(cfg.results_dir / f"base_output_{i}.png")
        ft_out.save(cfg.results_dir / f"finetuned_output_{i}.png")
        original.save(cfg.results_dir / f"original_{i}.png")
        mask.save(cfg.results_dir / f"mask_{i}.png")


if __name__ == "__main__":
    cfg = Config()
    _, eval_data = prepare_dataset(cfg)

    base_results = run_base_inference(cfg, eval_data)
    ft_results = run_finetuned_inference(cfg, eval_data)
    save_comparison(eval_data, base_results, ft_results, cfg)
    print("Inference complete!")
