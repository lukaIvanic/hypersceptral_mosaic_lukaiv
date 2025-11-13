from __future__ import annotations

import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = _make_group_norm(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = _make_group_norm(out_channels)
        if in_channels == out_channels:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        return self.act(out)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.pre = ResidualBlock(in_channels, out_channels, kernel_size=kernel_size)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pre(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = ResidualBlock(in_channels + skip_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


def _channel_multipliers(depth: int) -> List[int]:
    multipliers: List[int] = []
    value = 2
    for _ in range(max(depth, 1)):
        multipliers.append(value)
        if value < 8:
            value = min(value * 2, 8)
    return multipliers


class UNetLiteHSI(nn.Module):
    """
    UNet-style backbone operating in pixel-unshuffled space with aggressive
    stride-2 downsampling stages to handle native 1024×1024 mosaics efficiently.
    """

    variant_name = "unet_lite"

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 61,
        coarse_channels: int = 7,
        base_channels: int = 32,
        latent_channels: int = 32,
        encoder_depth: int = 3,
        train_resolution: int | None = None,
        pixel_factor: int = 2,
    ) -> None:
        super().__init__()
        if encoder_depth < 1:
            raise ValueError("encoder_depth must be >= 1.")
        if latent_channels < 1:
            raise ValueError("latent_channels must be >= 1.")
        if coarse_channels < 1 or coarse_channels > out_channels:
            raise ValueError("coarse_channels must be within [1, out_channels].")

        self.pixel_factor = pixel_factor
        self.out_channels = out_channels
        self.coarse_channels = coarse_channels
        self.train_resolution = train_resolution
        self.latent_channels = latent_channels
        self.encoder_depth = encoder_depth
        self.last_interp_time: float = 0.0

        stem_channels = in_channels * (pixel_factor**2)
        self.activation = nn.GELU()

        channel_progression = [base_channels]
        for mult in _channel_multipliers(encoder_depth):
            channel_progression.append(base_channels * mult)

        self.stem = nn.Sequential(
            nn.Conv2d(stem_channels, base_channels, kernel_size=3, padding=1),
            _make_group_norm(base_channels),
            nn.GELU(),
        )

        self.down_blocks = nn.ModuleList()
        for idx in range(encoder_depth):
            in_ch = channel_progression[idx]
            out_ch = channel_progression[idx + 1]
            self.down_blocks.append(DownBlock(in_ch, out_ch))

        bottleneck_channels = channel_progression[-1]
        self.bottleneck = ResidualBlock(bottleneck_channels, bottleneck_channels)

        skip_channels = channel_progression[1:]
        decoder_channels = list(reversed(channel_progression[:-1]))

        self.up_blocks = nn.ModuleList()
        current_channels = bottleneck_channels
        for out_ch, skip_ch in zip(decoder_channels, reversed(skip_channels)):
            self.up_blocks.append(UpBlock(current_channels, skip_ch, out_ch))
            current_channels = out_ch

        self.merge = ResidualBlock(current_channels + channel_progression[0], channel_progression[0])
        latent_unshuffle_channels = latent_channels * (pixel_factor**2)
        self.latent_proj = nn.Conv2d(channel_progression[0], latent_unshuffle_channels, kernel_size=1)
        self.latent_norm = _make_group_norm(latent_channels)
        self.coarse_head = nn.Conv2d(latent_channels, coarse_channels, kernel_size=1)

        self.required_divisor = pixel_factor * (2 ** encoder_depth)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _check_spatial_size(self, x: torch.Tensor) -> None:
        h, w = x.shape[-2:]
        if h % self.required_divisor != 0 or w % self.required_divisor != 0:
            raise ValueError(
                f"Input spatial dimensions must be divisible by {self.required_divisor}; got {(h, w)}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_spatial_size(x)
        x_unshuffled = F.pixel_unshuffle(x, self.pixel_factor)

        skip0 = self.stem(x_unshuffled)
        skips: List[torch.Tensor] = []
        x_encoded = skip0
        for block in self.down_blocks:
            x_encoded, skip = block(x_encoded)
            skips.append(skip)

        x_bottleneck = self.bottleneck(x_encoded)

        x_decoded = x_bottleneck
        for block in self.up_blocks:
            skip = skips.pop()
            x_decoded = block(x_decoded, skip)

        x_merged = torch.cat([x_decoded, skip0], dim=1)
        x_merged = self.merge(x_merged)

        latent_unshuffled = self.latent_proj(x_merged)
        latent = F.pixel_shuffle(latent_unshuffled, self.pixel_factor)
        latent = self.latent_norm(latent)
        latent = self.activation(latent)

        coarse = self.coarse_head(latent)
        coarse = coarse.permute(0, 2, 3, 1).contiguous()
        upsampled = F.interpolate(
            coarse.view(-1, 1, self.coarse_channels),
            size=self.out_channels,
            mode="linear",
            align_corners=True,
        )
        upsampled = upsampled.view(-1, coarse.shape[1], coarse.shape[2], self.out_channels)
        spectral = upsampled.permute(0, 3, 1, 2).contiguous()
        output = torch.sigmoid(spectral)
        self.last_interp_time = 0.0
        return output

    def predict_full_resolution(
        self,
        x: torch.Tensor,
        resize_to: int | tuple[int, int] | None = None,
        mode: str = "area",
        upsample_mode: str = "bilinear",
    ) -> torch.Tensor:
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
            small = F.interpolate(x, size=(target_train, target_train), mode=mode)
            total_interp += time.perf_counter() - start
        else:
            small = x

        pred_small = self.forward(small)

        if pred_small.shape[-2:] != (final_h, final_w):
            start = time.perf_counter()
            pred_small = F.interpolate(
                pred_small,
                size=(final_h, final_w),
                mode=upsample_mode,
                align_corners=False,
            )
            total_interp += time.perf_counter() - start

        self.last_interp_time = total_interp
        return pred_small


__all__ = ["UNetLiteHSI"]

