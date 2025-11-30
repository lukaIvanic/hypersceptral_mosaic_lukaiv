"""
Test-Time Augmentation (TTA) utilities.

Supports dihedral group D4 transforms (flips + 90° rotations) which form
8 unique augmentations. Predictions from each augmented input are
inverse-transformed and averaged for more robust outputs.

TTA Modes:
    none     - No augmentation (1 forward pass)
    flip     - Horizontal + vertical flips (4 forward passes)
    rotate90 - 90° rotations only (4 forward passes)
    dihedral - Full D4 group: flips + rotations (8 forward passes)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
from torch import nn

from .inference import run_model_with_resize


# Type alias for transform functions: (tensor) -> tensor
Transform = Callable[[torch.Tensor], torch.Tensor]


def identity(x: torch.Tensor) -> torch.Tensor:
    """Identity transform (no-op)."""
    return x


def flip_h(x: torch.Tensor) -> torch.Tensor:
    """Horizontal flip (flip along width axis)."""
    return torch.flip(x, dims=(-1,))


def flip_v(x: torch.Tensor) -> torch.Tensor:
    """Vertical flip (flip along height axis)."""
    return torch.flip(x, dims=(-2,))


def flip_hv(x: torch.Tensor) -> torch.Tensor:
    """Horizontal + vertical flip (equivalent to 180° rotation)."""
    return torch.flip(x, dims=(-2, -1))


def rot90_1(x: torch.Tensor) -> torch.Tensor:
    """Rotate 90° counter-clockwise."""
    return torch.rot90(x, k=1, dims=(-2, -1))


def rot90_2(x: torch.Tensor) -> torch.Tensor:
    """Rotate 180°."""
    return torch.rot90(x, k=2, dims=(-2, -1))


def rot90_3(x: torch.Tensor) -> torch.Tensor:
    """Rotate 270° counter-clockwise (90° clockwise)."""
    return torch.rot90(x, k=3, dims=(-2, -1))


def rot90_1_flip_h(x: torch.Tensor) -> torch.Tensor:
    """Rotate 90° then flip horizontally."""
    return flip_h(rot90_1(x))


def rot90_1_flip_v(x: torch.Tensor) -> torch.Tensor:
    """Rotate 90° then flip vertically."""
    return flip_v(rot90_1(x))


def rot90_3_flip_h(x: torch.Tensor) -> torch.Tensor:
    """Rotate 270° then flip horizontally."""
    return flip_h(rot90_3(x))


def rot90_3_flip_v(x: torch.Tensor) -> torch.Tensor:
    """Rotate 270° then flip vertically."""
    return flip_v(rot90_3(x))


# Inverse transforms for each forward transform
# For flips: same operation is its own inverse
# For rotations: inverse is rotation in opposite direction
INVERSE_TRANSFORMS: dict[Transform, Transform] = {
    identity: identity,
    flip_h: flip_h,
    flip_v: flip_v,
    flip_hv: flip_hv,
    rot90_1: rot90_3,  # inverse of 90° CCW is 90° CW (270° CCW)
    rot90_2: rot90_2,  # inverse of 180° is 180°
    rot90_3: rot90_1,  # inverse of 270° CCW is 90° CCW
    rot90_1_flip_h: lambda x: rot90_3(flip_h(x)),  # undo flip then undo rotation
    rot90_1_flip_v: lambda x: rot90_3(flip_v(x)),
    rot90_3_flip_h: lambda x: rot90_1(flip_h(x)),
    rot90_3_flip_v: lambda x: rot90_1(flip_v(x)),
}


# Pre-defined transform sets for each TTA mode
TTA_TRANSFORMS: dict[str, List[Transform]] = {
    "none": [identity],
    "flip": [identity, flip_h, flip_v, flip_hv],
    "rotate90": [identity, rot90_1, rot90_2, rot90_3],
    "dihedral": [
        identity,
        flip_h,
        flip_v,
        flip_hv,
        rot90_1,
        rot90_3,
        rot90_1_flip_h,
        rot90_3_flip_h,
    ],
}


def get_tta_transforms(mode: str) -> List[Transform]:
    """
    Get the list of transforms for a TTA mode.
    
    Args:
        mode: One of 'none', 'flip', 'rotate90', 'dihedral'
    
    Returns:
        List of transform functions
    
    Raises:
        ValueError: If mode is not recognized
    """
    mode_lower = mode.lower()
    if mode_lower not in TTA_TRANSFORMS:
        valid = ", ".join(sorted(TTA_TRANSFORMS.keys()))
        raise ValueError(f"Unknown TTA mode '{mode}'. Valid modes: {valid}")
    return TTA_TRANSFORMS[mode_lower]


def apply_tta(
    model: nn.Module,
    inputs: torch.Tensor,
    transforms: List[Transform],
    inference_resize: Optional[int] = None,
    final_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Apply test-time augmentation and average predictions.
    
    For each transform in the list:
    1. Apply transform to input
    2. Run model inference
    3. Apply inverse transform to prediction
    4. Accumulate predictions
    
    Finally, average all predictions.
    
    Args:
        model: Neural network model
        inputs: Input tensor of shape [B, C, H, W]
        transforms: List of transform functions to apply
        inference_resize: Optional resize for model inference
        final_shape: Optional target output shape (H, W)
    
    Returns:
        Averaged predictions of shape [B, C_out, H, W]
    """
    if not transforms:
        transforms = [identity]
    
    accumulated: Optional[torch.Tensor] = None
    num_augments = len(transforms)
    
    for transform in transforms:
        # Apply forward transform to input
        augmented_input = transform(inputs)
        
        # Run model inference
        pred = run_model_with_resize(
            model,
            augmented_input,
            inference_resize,
            final_shape,
        )
        
        # Apply inverse transform to prediction
        inverse = INVERSE_TRANSFORMS.get(transform, identity)
        pred_restored = inverse(pred)
        
        # Accumulate
        if accumulated is None:
            accumulated = pred_restored
        else:
            accumulated = accumulated + pred_restored
    
    # Average predictions
    if accumulated is None:
        raise RuntimeError("TTA produced no predictions")
    
    return accumulated / num_augments


def resolve_tta_mode(
    tta_mode: str,
    include_rotate90: bool = False,
) -> Tuple[str, List[Transform]]:
    """
    Resolve TTA mode string to actual transforms.
    
    Handles 'auto' mode which defaults to 'flip' but upgrades to 'dihedral'
    when include_rotate90 is True (matching training augmentation).
    
    Args:
        tta_mode: TTA mode string ('none', 'flip', 'rotate90', 'dihedral', 'auto')
        include_rotate90: Whether training used rotate90 augmentation
    
    Returns:
        Tuple of (resolved_mode_name, list_of_transforms)
    """
    mode_lower = tta_mode.lower()
    
    if mode_lower == "auto":
        # Auto mode: match training augmentation
        if include_rotate90:
            resolved = "dihedral"
        else:
            resolved = "flip"
    else:
        resolved = mode_lower
    
    transforms = get_tta_transforms(resolved)
    return resolved, transforms


__all__ = [
    "Transform",
    "identity",
    "flip_h",
    "flip_v", 
    "flip_hv",
    "rot90_1",
    "rot90_2",
    "rot90_3",
    "get_tta_transforms",
    "apply_tta",
    "resolve_tta_mode",
    "TTA_TRANSFORMS",
]

