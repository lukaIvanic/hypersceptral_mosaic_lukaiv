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


__all__ = ["run_model_with_resize"]


