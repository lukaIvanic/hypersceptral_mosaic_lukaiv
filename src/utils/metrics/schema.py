from __future__ import annotations

from typing import Dict, Iterable, Mapping

# Metric unit metadata ----------------------------------------------------- #

TRAINING_METRIC_UNITS: Dict[str, str] = {
    "loss": "1",
    "io": "s",
    "preprocess": "s",
    "forward": "s",
    "loss_fn": "s",
    "backward": "s",
    "optimizer": "s",
    "loss_item": "s",
    "interp": "s",
    "step_wall": "s",
    "ram_mb": "MB",
    "lr": "1",
}

ACCURACY_METRIC_UNITS: Dict[str, str] = {
    "loss": "1",
    "mae": "1",
    "mse": "1",
    "psnr": "dB",
    "psnr_hsi": "dB",
    "psnr_srgb": "dB",
    "sam": "deg",
    "sid": "nats",
    "ergas": "1",
    "ssim_srgb": "1",
    "deltae00": "dE00",
}

# Metric groupings --------------------------------------------------------- #

SPEED_METRICS = frozenset(
    {
        "io",
        "preprocess",
        "forward",
        "loss_fn",
        "backward",
        "optimizer",
        "loss_item",
        "interp",
        "step_wall",
        "ram_mb",
    }
)
ACCURACY_METRICS = frozenset(
    {
        "loss",
        "mae",
        "mse",
        "psnr",
        "psnr_hsi",
        "psnr_srgb",
        "sam",
        "sid",
        "ergas",
        "ssim_srgb",
        "deltae00",
    }
)


# Display guidance --------------------------------------------------------- #

MetricUnitsMap = Dict[str, str]


def units_from_defaults(
    keys: Iterable[str],
    defaults: Mapping[str, str],
    fallback: str = "1",
) -> MetricUnitsMap:
    """
    Build a units map for the provided metric keys using defaults where available.
    """

    return {key: defaults.get(key, fallback) for key in keys}


def resolution_label(resize_to: int | None) -> str:
    """
    Produce a compact label for the evaluated resolution.
    """

    if resize_to is None:
        return "native"
    return f"{resize_to}px"


def resolution_tag(resize_to: int | None) -> str:
    """
    Return a canonical tag for resolution-specific sections in the history JSON.
    """

    return "native" if resize_to is None else "resized"


SOURCE_TRAIN = "train"
SOURCE_EVALUATE = "evaluate"


METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "loss": "Loss",
    "mae": "MAE",
    "mse": "MSE",
    "psnr": "PSNR (HSI)",
    "psnr_hsi": "PSNR (HSI)",
    "psnr_srgb": "PSNR (sRGB)",
    "sam": "SAM",
    "sid": "SID",
    "ergas": "ERGAS",
    "ssim_srgb": "SSIM (sRGB)",
    "deltae00": "dE00",
    "io": "I/O Latency",
    "preprocess": "Preprocess Latency",
    "forward": "Forward Pass",
    "loss_fn": "Loss Eval",
    "backward": "Backward Pass",
    "optimizer": "Optimizer Step",
    "loss_item": "Loss Sync",
    "interp": "Interpolation",
    "step_wall": "Step Wall Time",
    "ram_mb": "RAM (avg)",
}

METRIC_GUIDANCE: Dict[str, Dict[str, float | str]] = {
    "loss": {"goal": "min"},
    "mae": {"goal": "min", "good": 0.02, "warn": 0.03},
    "mse": {"goal": "min", "good": 5e-4, "warn": 1e-3},
    "psnr": {"goal": "max", "good": 35.0, "warn": 30.0},
    "psnr_hsi": {"goal": "max", "good": 35.0, "warn": 30.0},
    "psnr_srgb": {"goal": "max", "good": 30.0, "warn": 25.0},
    "sam": {"goal": "min", "good": 7.0, "warn": 9.0},
    "sid": {"goal": "min", "good": 0.03, "warn": 0.05},
    "ergas": {"goal": "min", "good": 60.0, "warn": 80.0},
    "ssim_srgb": {"goal": "max", "good": 0.93, "warn": 0.9},
    "deltae00": {"goal": "min", "good": 4.0, "warn": 5.5},
}


