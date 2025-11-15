from __future__ import annotations

from typing import Optional

from torch import nn

from .simple_cnn.model import SimpleCNN
from .unet_lite import UNetLiteHSI


def create_model(
    variant: str,
    *,
    in_channels: int,
    out_channels: int,
    coarse_channels: int,
    hidden_channels: int,
    train_resolution: Optional[int],
    unet_base_channels: int,
    latent_channels: int,
    encoder_depth: int,
    use_residual_head: bool,
    use_spectral_conv: bool,
    spectral_conv_kernel_size: int,
    decoder_dropout: float,
    stochastic_depth_p: float,
    use_bottleneck_attention: bool,
) -> nn.Module:
    variant_norm = variant.lower()
    if variant_norm == "baseline":
        return SimpleCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            coarse_channels=coarse_channels,
            hidden_channels=hidden_channels,
            train_resolution=train_resolution,
        )
    if variant_norm == "unet_lite":
        return UNetLiteHSI(
            in_channels=in_channels,
            out_channels=out_channels,
            coarse_channels=coarse_channels,
            base_channels=unet_base_channels,
            latent_channels=latent_channels,
            encoder_depth=encoder_depth,
            train_resolution=train_resolution,
            use_residual_head=use_residual_head,
            use_spectral_conv=use_spectral_conv,
            spectral_conv_kernel_size=spectral_conv_kernel_size,
            decoder_dropout=decoder_dropout,
            stochastic_depth_p=stochastic_depth_p,
            use_bottleneck_attention=use_bottleneck_attention,
        )
    raise ValueError(f"Unknown model variant '{variant}'. Supported: baseline, unet_lite.")

