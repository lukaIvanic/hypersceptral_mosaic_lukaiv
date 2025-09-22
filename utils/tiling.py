from __future__ import annotations
import math
from typing import Tuple

import numpy as np
import torch


def _hann2d(h: int, w: int, eps: float = 1e-6) -> torch.Tensor:
    """2D Hann window in [0,1], outer product of 1D windows."""
    wy = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(h) / max(h - 1, 1)))
    wx = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(w) / max(w - 1, 1)))
    win = np.outer(wy, wx).astype(np.float32)
    win = win / (win.max() + eps)
    return torch.from_numpy(win)


@torch.no_grad()
def predict_tiled(
    model: torch.nn.Module,
    x: torch.Tensor,
    tile: int = 512,
    overlap: int = 64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Predict on large images by tiling with overlap + Hann blending.

    Args
    ----
    model : nn.Module mapping (N,C,H,W)->(N,Cout,H,W)
    x     : (1,C,H,W) input tensor
    tile  : tile size (square tiles)
    overlap : pixels overlapped between adjacent tiles
    device : torch.device

    Returns
    -------
    y : (1,Cout,H,W)
    """
    assert x.ndim == 4 and x.size(0) == 1
    _, c, H, W = x.shape
    step = max(tile - overlap, 1)
    win2d = _hann2d(tile, tile).to(device)

    # Lazy run one center tile to get Cout
    r0 = min(0, H - tile)
    c0 = min(0, W - tile)
    tile0 = x[:, :, 0: min(tile, H), 0: min(tile, W)].to(device)
    y0 = model(tile0).clamp(0, 1)
    _, cout, _, _ = y0.shape

    out = torch.zeros((1, cout, H, W), dtype=y0.dtype, device=device)
    norm = torch.zeros((1, 1, H, W), dtype=y0.dtype, device=device)

    for r in range(0, H, step):
        r1 = min(r + tile, H); rr = r1 - r
        for c_ in range(0, W, step):
            c1 = min(c_ + tile, W); cc = c1 - c_
            x_tile = x[:, :, r:r1, c_:c1].to(device)
            y_tile = model(x_tile).clamp(0, 1)
            # Window for current actual tile size
            w2 = _hann2d(rr, cc).to(device)
            w2 = w2[None, None, :, :]
            out[:, :, r:r1, c_:c1] += y_tile * w2
            norm[:, :, r:r1, c_:c1] += w2

    out = out / torch.clamp(norm, min=1e-8)
    return out

