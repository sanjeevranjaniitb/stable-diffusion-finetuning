"""
Download and prepare a small dataset for inpainting fine-tuning.
Creates image + mask pairs from a public HuggingFace dataset.
"""

import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw
from torchvision import transforms

from config import Config


def generate_random_mask(image_size: int, seed: int = None) -> Image.Image:
    """Generate a random rectangular mask for inpainting."""
    if seed is not None:
        random.seed(seed)

    mask = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask)

    # Random rectangle covering 15-35% of the image
    min_frac, max_frac = 0.15, 0.35
    w = random.randint(int(image_size * min_frac), int(image_size * max_frac))
    h = random.randint(int(image_size * min_frac), int(image_size * max_frac))
    x = random.randint(0, image_size - w)
    y = random.randint(0, image_size - h)
    draw.rectangle([x, y, x + w, y + h], fill=255)

    return mask


def prepare_dataset(cfg: Config):
    """Download dataset and create train/eval splits with masks."""
    print("Downloading dataset from HuggingFace...")
    ds = load_dataset(cfg.dataset_name, split="train")
    print(f"  Downloaded {len(ds)} images")

    # Limit samples
    total_needed = cfg.num_train_samples + cfg.num_eval_samples
    if len(ds) < total_needed:
        # Duplicate images if dataset is too small
        print(f"  Dataset has {len(ds)} images, need {total_needed}. Duplicating...")
        indices = list(range(len(ds))) * (total_needed // len(ds) + 1)
        indices = indices[:total_needed]
    else:
        indices = list(range(total_needed))

    random.seed(cfg.seed)
    random.shuffle(indices)

    resize = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.CenterCrop(cfg.image_size),
    ])

    train_data = []
    eval_data = []

    for i, idx in enumerate(indices):
        item = ds[idx % len(ds)]
        image = item["image"].convert("RGB")
        image = resize(image)
        mask = generate_random_mask(cfg.image_size, seed=cfg.seed + i)

        # Create masked image (zero out masked region)
        masked_image = image.copy()
        mask_np = np.array(mask) / 255.0
        img_np = np.array(masked_image).astype(float)
        for c in range(3):
            img_np[:, :, c] *= (1.0 - mask_np)
        masked_image = Image.fromarray(img_np.astype(np.uint8))

        entry = {
            "image": image,
            "mask": mask,
            "masked_image": masked_image,
            "index": i,
        }

        if i < cfg.num_train_samples:
            train_data.append(entry)
        else:
            eval_data.append(entry)

    print(f"  Prepared {len(train_data)} train, {len(eval_data)} eval samples")
    return train_data, eval_data


class InpaintingDataset(torch.utils.data.Dataset):
    """PyTorch dataset for inpainting training."""

    def __init__(self, data: list, image_size: int):
        self.data = data
        self.image_size = image_size
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # scale to [-1, 1]
        ])
        self.mask_to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = self.to_tensor(entry["image"])
        mask = self.mask_to_tensor(entry["mask"])
        masked_image = self.to_tensor(entry["masked_image"])
        return {
            "image": image,
            "mask": mask,
            "masked_image": masked_image,
        }


if __name__ == "__main__":
    cfg = Config()
    train_data, eval_data = prepare_dataset(cfg)

    # Save sample images for inspection
    sample_dir = cfg.output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    for i, entry in enumerate(train_data[:3]):
        entry["image"].save(sample_dir / f"train_{i}_original.png")
        entry["mask"].save(sample_dir / f"train_{i}_mask.png")
        entry["masked_image"].save(sample_dir / f"train_{i}_masked.png")
    print(f"Sample images saved to {sample_dir}")
