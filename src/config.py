from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainConfig:
    """
    Minimal training hyperparameters and paths.

    Feel free to tweak the defaults or expose new fields via argparse in
    ``train.py``. The intent is to keep a single source of truth for shapes and
    core hyperparameters that both training and evaluation can reuse.
    """

    data_root: Path = Path("data/track1")
    run_name: str = "demo-run"

    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    num_workers: int = 0
    prefetch_factor: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Model shape defaults
    input_channels: int = 1  # mosaics are single channel
    output_channels: int = 61  # hyperspectral cube bands
    coarse_output_channels: int = 7  # channels predicted before spectral upsampling
    hidden_channels: int = 32

    # Optional dataset spatial resize (None keeps native resolution)
    resize_to: int | None = None
    # Optional inference resize applied inside the training loop before model forward
    train_inference_resize: int | None = None
    cache_dir: Path | None = Path("data/cache/track1")
    write_cache: bool = True
    ram_cache: bool = False

    # Logging/checkpointing cadence
    log_interval: int = 20
    val_interval: int = 1
    checkpoint_every: int = 1

    # Loss weights (Step 1A defaults)
    lambda_l1: float = 1.0
    lambda_sam: float = 0.1
    lambda_sid: float = 0.0
    lambda_srgb_l1: float = 0.0
    lambda_srgb_ssim: float = 0.0
    lambda_ergas: float = 0.0  # applied to normalized ERGAS (≈0–1 range)

    # Model variants
    model_variant: str = "baseline"
    unet_base_channels: int = 32
    latent_channels: int = 32
    encoder_depth: int = 3

    # LR scheduler
    lr_scheduler: str = "none"  # options: none, cosine
    scheduler_warmup_epochs: int = 0
    scheduler_warmup_start_factor: float = 0.1
    scheduler_min_lr: float = 1e-5


__all__ = ["TrainConfig"]


