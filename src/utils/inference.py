from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


def run_model_with_resize(
    model: nn.Module,
    inputs: torch.Tensor,
    inference_resize: Optional[int],
    final_shape: Optional[Tuple[int, int]],
) -> torch.Tensor:
    """
    Downsample inputs before the forward pass (if requested) and optionally upsample
    model outputs back to a target spatial size.
    """
    want_downsample = (
        inference_resize is not None
        and (inputs.shape[-2] != inference_resize or inputs.shape[-1] != inference_resize)
    )
    want_final = final_shape is not None and inputs.shape[-2:] != final_shape
    if hasattr(model, "predict_full_resolution") and (want_downsample or want_final):
        resize_arg = final_shape if final_shape is not None else None
        return model.predict_full_resolution(inputs, resize_to=resize_arg)
    return model(inputs)


def sliding_window_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    patch_size: int,
    overlap: float,
    inference_resize: Optional[int],
) -> torch.Tensor:
    """
    Run the model on an image using sliding-window inference.

    Args:
        model: Trained model to evaluate.
        input_tensor: Tensor of shape (C, H, W).
        patch_size: Square crop size.
        overlap: Fractional overlap between 0.0 and <1.0.
        inference_resize: Optional resize hint passed through to ``run_model_with_resize``.

    Returns:
        Tensor of shape (C_out, H, W) with averaged predictions over overlapping patches.
    """

    if input_tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {input_tensor.shape}")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("overlap must be in the range [0.0, 1.0).")

    _, height, width = input_tensor.shape
    if patch_size > height or patch_size > width:
        raise ValueError(
            f"patch_size={patch_size} exceeds input dimensions ({height}, {width})."
        )

    stride = patch_size
    if overlap > 0.0:
        stride = max(1, int(round(patch_size * (1.0 - overlap))))
    if stride > patch_size:
        stride = patch_size

    def _generate_starts(size: int, window: int, step: int) -> list[int]:
        starts: list[int] = []
        current = 0
        while True:
            starts.append(current)
            if current + window >= size:
                break
            current += step
            if current + window > size:
                current = size - window
        return starts

    rows = _generate_starts(height, patch_size, stride)
    cols = _generate_starts(width, patch_size, stride)

    output_tensor: Optional[torch.Tensor] = None
    accumulation: Optional[torch.Tensor] = None

    for top in rows:
        bottom = top + patch_size
        for left in cols:
            right = left + patch_size
            crop = input_tensor[:, top:bottom, left:right].unsqueeze(0)
            pred_patch = run_model_with_resize(
                model,
                crop,
                inference_resize,
                (patch_size, patch_size),
            ).squeeze(0)

            if output_tensor is None:
                output_tensor = torch.zeros(
                    (pred_patch.shape[0], height, width),
                    dtype=pred_patch.dtype,
                    device=pred_patch.device,
                )
                accumulation = torch.zeros(
                    (1, height, width),
                    dtype=pred_patch.dtype,
                    device=pred_patch.device,
                )

            output_tensor[:, top:bottom, left:right] += pred_patch
            accumulation[:, top:bottom, left:right] += 1.0

    if output_tensor is None or accumulation is None:
        raise RuntimeError("Sliding window inference produced no patches.")

    output_tensor = output_tensor / accumulation.clamp_min(1.0)
    return output_tensor


__all__ = ["run_model_with_resize", "sliding_window_inference"]


