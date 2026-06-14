from __future__ import annotations

from typing import Tuple

import torch

DELTA_LAMBDA = 10.0  # nm spacing between bands

_WAVELENGTHS = torch.arange(400.0, 1001.0, DELTA_LAMBDA)

# CIE 1931 2° color matching functions sampled at 10 nm (400–780 nm), zeros beyond.
_CMF_X_BASE = [
    0.01431,
    0.04351,
    0.13438,
    0.28390,
    0.34828,
    0.33620,
    0.29080,
    0.19536,
    0.09564,
    0.03201,
    0.00490,
    0.00930,
    0.06327,
    0.16550,
    0.29040,
    0.43345,
    0.59450,
    0.76210,
    0.91630,
    1.02630,
    1.06220,
    1.00260,
    0.85445,
    0.64240,
    0.44790,
    0.28350,
    0.16490,
    0.08740,
    0.04677,
    0.02270,
    0.01136,
    0.00579,
    0.00290,
    0.00144,
    0.00069,
    0.00032,
    0.00015,
    0.00007,
    0.00003,
]
_CMF_Y_BASE = [
    0.000396,
    0.001210,
    0.004000,
    0.011600,
    0.023000,
    0.038000,
    0.060000,
    0.091000,
    0.139020,
    0.208020,
    0.323000,
    0.503000,
    0.710000,
    0.862000,
    0.954000,
    0.995000,
    0.995000,
    0.952000,
    0.870000,
    0.757000,
    0.631000,
    0.503000,
    0.381000,
    0.265000,
    0.175000,
    0.107000,
    0.061000,
    0.032000,
    0.017000,
    0.008210,
    0.004102,
    0.002091,
    0.001048,
    0.000520,
    0.000249,
    0.000120,
    0.000060,
    0.000030,
    0.000015,
]
_CMF_Z_BASE = [
    0.067850,
    0.207400,
    0.645600,
    1.385600,
    1.747060,
    1.772110,
    1.669200,
    1.287640,
    0.813000,
    0.465180,
    0.272000,
    0.158200,
    0.078250,
    0.042160,
    0.020300,
    0.008750,
    0.003900,
    0.002100,
    0.001050,
    0.000520,
    0.000250,
    0.000120,
    0.000060,
    0.000030,
    0.000020,
    0.000010,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
    0.000000,
]

# Extend CMFs with zeros to 1000 nm (61 bands total).
_EXTRA_ZEROS = [0.0] * (len(_WAVELENGTHS) - len(_CMF_X_BASE))
_CMF_X = torch.tensor(_CMF_X_BASE + _EXTRA_ZEROS, dtype=torch.float32)
_CMF_Y = torch.tensor(_CMF_Y_BASE + _EXTRA_ZEROS, dtype=torch.float32)
_CMF_Z = torch.tensor(_CMF_Z_BASE + _EXTRA_ZEROS, dtype=torch.float32)

# Relative SPD of standard illuminant D65 at 10 nm sampling (400–780 nm), zeros beyond.
_D65_BASE = [
    82.7549,
    91.4860,
    93.4318,
    86.6823,
    104.865,
    117.008,
    117.812,
    114.861,
    115.923,
    108.811,
    109.354,
    107.802,
    104.790,
    107.689,
    104.405,
    104.046,
    100.000,
    96.3342,
    95.7880,
    88.6856,
    90.0062,
    89.5991,
    87.6987,
    83.2886,
    83.6992,
    80.0268,
    80.2146,
    82.2778,
    78.2842,
    69.7213,
    71.6091,
    74.3490,
    61.6040,
    65.7448,
    63.3828,
    55.3296,
    58.8765,
    61.0000,
    57.4589,
]
_D65 = torch.tensor(_D65_BASE + _EXTRA_ZEROS, dtype=torch.float32)

_CMFS = torch.stack([_CMF_X, _CMF_Y, _CMF_Z], dim=0)  # (3, 61)

_NORMALISATION = 1.0 / (torch.sum(_CMF_Y * _D65) * DELTA_LAMBDA)
_XYZ_WEIGHTS = (_CMFS * _D65.unsqueeze(0)) * (DELTA_LAMBDA * _NORMALISATION)
_XYZ_WEIGHTS = _XYZ_WEIGHTS.transpose(0, 1).contiguous()  # (61, 3)

_WHITE_XYZ = torch.sum(_XYZ_WEIGHTS, dim=0)  # XYZ of perfect diffuser (Y ≈ 1)

_XYZ_TO_SRGB = torch.tensor(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=torch.float32,
)


def _ensure_tensor_on_device(t: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if t.device == device and t.dtype == dtype:
        return t
    return t.to(device=device, dtype=dtype)


def _reshape_spectral(cube: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], bool]:
    if cube.dim() == 4:
        b, c, h, w = cube.shape
        spectra = cube.permute(0, 2, 3, 1).reshape(-1, c)
        leading = (b, h, w)
        batched = True
    elif cube.dim() == 3:
        c, h, w = cube.shape
        spectra = cube.permute(1, 2, 0).reshape(-1, c)
        leading = (h, w)
        batched = False
    else:
        raise ValueError(f"Expected tensor of shape (B,C,H,W) or (C,H,W), got {cube.shape}")
    return spectra, leading, batched


def _restore_tristimulus(values: torch.Tensor, leading: Tuple[int, ...], batched: bool) -> torch.Tensor:
    if batched:
        b, h, w = leading
        return values.reshape(b, h, w, 3).permute(0, 3, 1, 2).contiguous()
    h, w = leading
    return values.reshape(h, w, 3).permute(2, 0, 1).contiguous()


def spectral_to_xyz(cube: torch.Tensor) -> torch.Tensor:
    spectra, leading, batched = _reshape_spectral(cube)
    device = cube.device
    dtype = cube.dtype
    weights = _ensure_tensor_on_device(_XYZ_WEIGHTS, device, dtype)
    xyz = spectra @ weights
    return _restore_tristimulus(xyz, leading, batched)


def xyz_to_linear_srgb(xyz: torch.Tensor) -> torch.Tensor:
    if xyz.dim() not in (3, 4):
        raise ValueError(f"Expected XYZ tensor of shape (B,3,H,W) or (3,H,W), got {xyz.shape}")
    device = xyz.device
    dtype = xyz.dtype
    matrix = _ensure_tensor_on_device(_XYZ_TO_SRGB, device, dtype)
    if xyz.dim() == 4:
        b, _, h, w = xyz.shape
        flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
        rgb = flat @ matrix.T
        return rgb.reshape(b, h, w, 3).permute(0, 3, 1, 2).contiguous()
    # (3, H, W)
    h, w = xyz.shape[1:]
    flat = xyz.permute(1, 2, 0).reshape(-1, 3)
    rgb = flat @ matrix.T
    return rgb.reshape(h, w, 3).permute(2, 0, 1).contiguous()


def linear_to_srgb(linear_rgb: torch.Tensor) -> torch.Tensor:
    threshold = 0.0031308
    below = linear_rgb <= threshold
    above = ~below
    encoded = torch.zeros_like(linear_rgb)
    encoded[below] = linear_rgb[below] * 12.92
    encoded[above] = 1.055 * torch.pow(torch.clamp(linear_rgb[above], min=0.0), 1 / 2.4) - 0.055
    return encoded


def spectral_to_srgb(cube: torch.Tensor, apply_gamma: bool = True) -> torch.Tensor:
    xyz = spectral_to_xyz(cube)
    linear = torch.clamp(xyz_to_linear_srgb(xyz), min=0.0)
    return torch.clamp(linear_to_srgb(linear) if apply_gamma else linear, min=0.0, max=1.0)


def spectral_to_lab(cube: torch.Tensor) -> torch.Tensor:
    xyz = spectral_to_xyz(cube)
    return xyz_to_lab(xyz)


def xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    if xyz.dim() not in (3, 4):
        raise ValueError(f"Expected XYZ tensor of shape (B,3,H,W) or (3,H,W), got {xyz.shape}")
    device = xyz.device
    dtype = xyz.dtype
    white = _ensure_tensor_on_device(_WHITE_XYZ, device, dtype)

    if xyz.dim() == 4:
        batch_size, _, h, w = xyz.shape
        flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
    else:
        h, w = xyz.shape[1:]
        batch_size = None
        flat = xyz.permute(1, 2, 0).reshape(-1, 3)

    ratios = flat / white
    delta = (6.0 / 29.0)
    delta3 = delta ** 3
    mask = ratios > delta3
    f = torch.where(mask, ratios.pow(1.0 / 3.0), (ratios / (3 * delta * delta)) + (4.0 / 29.0))

    L = (116.0 * f[:, 1]) - 16.0
    a_star = 500.0 * (f[:, 0] - f[:, 1])
    b_star = 200.0 * (f[:, 1] - f[:, 2])
    lab = torch.stack((L, a_star, b_star), dim=1)

    if xyz.dim() == 4:
        return lab.reshape(batch_size, h, w, 3).permute(0, 3, 1, 2).contiguous()
    return lab.reshape(h, w, 3).permute(2, 0, 1).contiguous()


__all__ = [
    "DELTA_LAMBDA",
    "spectral_to_xyz",
    "spectral_to_srgb",
    "spectral_to_lab",
    "xyz_to_lab",
    "xyz_to_linear_srgb",
    "linear_to_srgb",
]
