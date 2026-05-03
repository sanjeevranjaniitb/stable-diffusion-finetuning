"""
Configuration for the SD inpainting fine-tuning demo.
Tuned for 24GB Mac RAM with MPS backend.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Model
    model_id: str = "runwayml/stable-diffusion-inpainting"
    revision: str = "fp16"
    dtype: str = "float32"  # MPS works best with float32

    # LoRA parameters (keep small for demo)
    lora_rank: int = 4
    lora_alpha: int = 4
    lora_dropout: float = 0.0
    lora_target_modules: list = field(
        default_factory=lambda: ["to_q", "to_v", "to_k", "to_out.0"]
    )

    # Dataset
    dataset_name: str = "diffusers/dog-example"  # tiny HF dataset
    num_train_samples: int = 5
    num_eval_samples: int = 2
    image_size: int = 256  # smaller for speed on Mac

    # Training
    num_epochs: int = 10
    train_batch_size: int = 1
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    max_train_steps: int = 50  # cap total steps for demo
    seed: int = 42

    # Paths
    output_dir: Path = Path("outputs")
    lora_weights_dir: Path = Path("outputs/lora_weights")
    results_dir: Path = Path("outputs/results")
    eval_dir: Path = Path("outputs/evaluation")

    # Inference
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    inpaint_prompt: str = "a dog sitting on a bench in a park, high quality photo"

    def __post_init__(self):
        for d in [self.output_dir, self.lora_weights_dir, self.results_dir, self.eval_dir]:
            d.mkdir(parents=True, exist_ok=True)
