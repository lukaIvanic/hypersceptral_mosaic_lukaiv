from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


@dataclass
class PCABasis:
    mean: np.ndarray   # (C,)
    components: np.ndarray  # (K,C) PCA loading vectors; rows are components

    def to_torch(self, device: torch.device) -> "PCABasisTorch":
        return PCABasisTorch(
            mean=torch.from_numpy(self.mean.astype(np.float32)).to(device),
            components=torch.from_numpy(self.components.astype(np.float32)).to(device),
        )


@dataclass
class PCABasisTorch:
    mean: torch.Tensor       # (C,)
    components: torch.Tensor # (K,C)


class PCACoeffMapper(nn.Module):
    """
    Predicts PCA coefficients per pixel from mosaic 2×2 packed input.

    - PixelUnshuffle(2): (N,1,H,W) -> (N,4,H/2,W/2)
    - 1x1 Conv: 4 -> (K*4)
    - PixelShuffle(2): -> (N,K,H,W) coefficients per pixel
    - Reconstruct 61 bands via fixed PCA basis: X ≈ mean + coeffs @ components

    components: (K,C), mean: (C,)
    """
    def __init__(self, basis: PCABasisTorch, out_bands: int = 61):
        super().__init__()
        K, C = int(basis.components.shape[0]), int(basis.components.shape[1])
        assert C == out_bands, f"Basis bands {C} != out_bands {out_bands}"
        self.basis = basis
        self.unshuffle = nn.PixelUnshuffle(2)
        self.head = nn.Conv2d(4, K * 4, kernel_size=1, stride=1, padding=0, bias=True)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, mosaic: torch.Tensor) -> torch.Tensor:
        x = self.unshuffle(mosaic)                # (N,4,H/2,W/2)
        k4 = self.head(x)                         # (N,K*4,H/2,W/2)
        coeffs = self.shuffle(k4)                 # (N,K,H,W)
        # Reconstruct: mean + coeffs @ components
        # reshape to (N,H,W,K) @ (K,C) -> (N,H,W,C)
        N, K, H, W = coeffs.shape
        coeffs_nhwk = coeffs.permute(0, 2, 3, 1).reshape(-1, K)
        C = self.basis.components.shape[1]
        rec = coeffs_nhwk @ self.basis.components  # (N*H*W, C)
        rec = rec + self.basis.mean[None, :]
        rec = rec.reshape(N, H, W, C).permute(0, 3, 1, 2)
        return rec.clamp(0.0, 1.0)

