"""
Fine-tune Stable Diffusion Inpainting with LoRA on a small dataset.
Optimized for Apple Silicon Mac with MPS backend.
"""

import gc
import time

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from config import Config
from data_prep import InpaintingDataset, prepare_dataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(cfg: Config, device: torch.device):
    """Load SD inpainting model components."""
    print("Loading model components...")

    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.model_id, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.model_id, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.model_id, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.model_id, subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.model_id, subfolder="scheduler"
    )

    # Freeze everything except UNet (which gets LoRA)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # Move frozen models to device
    text_encoder.to(device)
    vae.to(device)

    print("  Models loaded successfully")
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def apply_lora(unet, cfg: Config):
    """Apply LoRA adapters to the UNet."""
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
    )
    unet = get_peft_model(unet, lora_config)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"  LoRA applied: {trainable:,} trainable / {total:,} total params "
          f"({100 * trainable / total:.2f}%)")
    return unet


def encode_prompt(prompt: str, tokenizer, text_encoder, device):
    """Encode a text prompt into CLIP embeddings."""
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeddings = text_encoder(tokens.input_ids.to(device))[0]
    return embeddings


def train(cfg: Config):
    """Run the LoRA fine-tuning loop."""
    device = get_device()
    print(f"Using device: {device}")

    # Load models
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(cfg, device)

    # Apply LoRA
    unet = apply_lora(unet, cfg)
    unet.to(device)

    # Prepare data
    train_data, eval_data = prepare_dataset(cfg)
    train_dataset = InpaintingDataset(train_data, cfg.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
    )

    # Encode the training prompt
    prompt_embeds = encode_prompt(
        cfg.inpaint_prompt, tokenizer, text_encoder, device
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=cfg.learning_rate,
    )

    # Training loop
    print(f"\nStarting training: {cfg.max_train_steps} steps")
    print(f"  Epochs: {cfg.num_epochs}, Batch size: {cfg.train_batch_size}")
    print(f"  Image size: {cfg.image_size}x{cfg.image_size}")
    print()

    global_step = 0
    losses = []
    start_time = time.time()

    for epoch in range(cfg.num_epochs):
        unet.train()
        epoch_loss = 0.0

        for batch in train_loader:
            if global_step >= cfg.max_train_steps:
                break

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            masked_images = batch["masked_image"].to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                masked_latents = vae.encode(masked_images).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

            # Resize mask to latent size
            mask_latents = F.interpolate(
                masks, size=latents.shape[-2:], mode="nearest"
            )

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Concatenate for inpainting: noisy_latents + mask + masked_latents
            # SD inpainting UNet expects 9-channel input
            unet_input = torch.cat([noisy_latents, mask_latents, masked_latents], dim=1)

            # Expand prompt embeddings for batch
            encoder_hidden_states = prompt_embeds.expand(images.shape[0], -1, -1)

            # Predict noise
            noise_pred = unet(
                unet_input, timesteps, encoder_hidden_states
            ).sample

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            epoch_loss += loss_val
            losses.append(loss_val)
            global_step += 1

            if global_step % 10 == 0 or global_step == 1:
                elapsed = time.time() - start_time
                print(f"  Step {global_step}/{cfg.max_train_steps} | "
                      f"Loss: {loss_val:.4f} | "
                      f"Time: {elapsed:.1f}s")

        if global_step >= cfg.max_train_steps:
            break

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}/{cfg.num_epochs} | Avg Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}")

    # Save LoRA weights
    unet.save_pretrained(str(cfg.lora_weights_dir))
    print(f"  LoRA weights saved to {cfg.lora_weights_dir}")

    # Cleanup
    del unet, vae, text_encoder, optimizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return losses, eval_data


if __name__ == "__main__":
    cfg = Config()
    losses, _ = train(cfg)
    print(f"Training finished. Loss went from {losses[0]:.4f} to {losses[-1]:.4f}")
