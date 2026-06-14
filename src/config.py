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

    # Optional spatial resize (None keeps native resolution)
    resize_to: int | None = 64
    cache_dir: Path | None = Path("data/cache/track1")
    write_cache: bool = True

    # Logging/checkpointing cadence
    log_interval: int = 20
    val_interval: int = 1
    checkpoint_every: int = 1


__all__ = ["TrainConfig"]
