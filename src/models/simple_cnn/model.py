from __future__ import annotations

import torch
import time
from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Extremely small baseline for mosaic → HSI reconstruction.

    Architecture:
        Conv3×3(in → hidden) + ReLU
        Conv3×3(hidden → hidden) + ReLU
        Conv1×1(hidden → 61 bands)

    The model clamps the output to [0, 1] to match reflectance bounds.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 61,
        coarse_channels: int = 7,
        hidden_channels: int = 32,
        train_resolution: int | None = None,
    ) -> None:
        super().__init__()
        self.coarse_channels = coarse_channels
        self.out_channels = out_channels
        self.train_resolution = train_resolution

        self.last_interp_time: float = 0.0

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, coarse_channels, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse = torch.clamp(self.net(x), 0.0, 1.0)
        if self.coarse_channels == self.out_channels:
            self.last_interp_time = 0.0
            return coarse

        b, c, h, w = coarse.shape
        interp_start = time.perf_counter()
        # reshape to (N, C=1, L=coarse_channels) for 1D interpolation along spectra
        reshaped = coarse.permute(0, 2, 3, 1).reshape(-1, 1, self.coarse_channels)
        upsampled = F.interpolate(
            reshaped,
            size=self.out_channels,
            mode="linear",
            align_corners=True,
        )
        self.last_interp_time = time.perf_counter() - interp_start
        upsampled = upsampled.reshape(b, h, w, self.out_channels).permute(0, 3, 1, 2)
        return torch.clamp(upsampled, 0.0, 1.0)

    def predict_full_resolution(
        self,
        x: torch.Tensor,
        resize_to: int | None = None,
        mode: str = "area",
    ) -> torch.Tensor:
        """
        Run inference on native-resolution mosaics by resizing to the training
        resolution, invoking the network, and upsampling back.
        """
        if resize_to is None:
            resize_to = self.train_resolution
        if resize_to is None:
            return self.forward(x)

        orig_h, orig_w = x.shape[-2:]
        if orig_h == resize_to and orig_w == resize_to:
            return self.forward(x)

        small = F.interpolate(x, size=(resize_to, resize_to), mode=mode)
        pred_small = self.forward(small)
        pred_full = F.interpolate(pred_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return pred_full


# Backwards-compatibility alias for earlier imports
SimpleHSIModel = SimpleCNN

__all__ = ["SimpleCNN", "SimpleHSIModel"]
