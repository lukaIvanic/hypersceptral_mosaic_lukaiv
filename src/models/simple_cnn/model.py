from __future__ import annotations

import torch
import time
from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Extremely small baseline for mosaic → HSI reconstruction.

    Pipeline:
        1. Pixel-unshuffle RGGB mosaic (factor 2) → 4 channels at half resolution.
        2. Three 5×5 conv layers (hidden=32) operating in the unshuffled space.
        3. Pixel-shuffle back to native resolution, then optional spectral interpolation.

    The model clamps the output to [0, 1] to match reflectance bounds.
    """

    variant_name = "baseline"

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
        self._pixel_factor = 2
        self.variant_name = "baseline"

        self.last_interp_time: float = 0.0

        input_channels = in_channels * (self._pixel_factor ** 2)
        output_channels = coarse_channels * (self._pixel_factor ** 2)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self._pixel_factor != 0 or x.shape[-2] % self._pixel_factor != 0:
            raise ValueError(
                f"Input spatial dimensions must be divisible by {self._pixel_factor}; "
                f"got {tuple(x.shape[-2:])}."
            )
        x = F.pixel_unshuffle(x, self._pixel_factor)

        coarse = self.net(x)

        coarse = F.pixel_shuffle(coarse, self._pixel_factor)

        coarse = torch.clamp(coarse, 0.0, 1.0)
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
        resize_to: int | tuple[int, int] | None = None,
        mode: str = "area",
        upsample_mode: str = "bilinear",
    ) -> torch.Tensor:
        """
        Run inference by resizing to the training resolution, invoking the
        network, and (optionally) upsampling the prediction to a target size.
        """
        orig_h, orig_w = x.shape[-2:]
        if isinstance(resize_to, int):
            final_h = final_w = resize_to
        elif isinstance(resize_to, tuple):
            if len(resize_to) != 2:
                raise ValueError("resize_to tuple must have length 2.")
            final_h, final_w = int(resize_to[0]), int(resize_to[1])
        else:
            final_h, final_w = orig_h, orig_w

        target_train = self.train_resolution
        total_interp = 0.0

        if target_train is not None and (orig_h != target_train or orig_w != target_train):
            start = time.perf_counter()
            inp = F.interpolate(x, size=(target_train, target_train), mode=mode)
            total_interp += time.perf_counter() - start
        else:
            inp = x

        pred = self.forward(inp)

        if pred.shape[-2:] != (final_h, final_w):
            start = time.perf_counter()
            pred = F.interpolate(pred, size=(final_h, final_w), mode=upsample_mode, align_corners=False)
            total_interp += time.perf_counter() - start

        self.last_interp_time = total_interp
        return pred


# Backwards-compatibility alias for earlier imports
SimpleHSIModel = SimpleCNN

__all__ = ["SimpleCNN", "SimpleHSIModel"]

