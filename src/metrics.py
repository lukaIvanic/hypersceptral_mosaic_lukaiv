from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Tuple

import torch
import torch.nn.functional as F

from .color import (
    linear_to_srgb,
    spectral_to_lab,
    spectral_to_srgb,
    spectral_to_xyz,
    xyz_to_lab,
    _XYZ_TO_SRGB,
    _XYZ_WEIGHTS,
)

MetricFn = Callable[[torch.Tensor, torch.Tensor, MutableMapping[str, torch.Tensor]], float]

_EPS = 1e-8

_FAST_WEIGHTS: Dict[Tuple[str, Optional[int], torch.dtype], Dict[str, torch.Tensor]] = {}
_XYZ_CONV_WEIGHT_BASE = _XYZ_WEIGHTS.t().contiguous().view(3, _XYZ_WEIGHTS.shape[0], 1, 1)
_SRGB_CONV_WEIGHT_BASE = _XYZ_TO_SRGB.contiguous().view(3, 3, 1, 1)


def _device_key(tensor: torch.Tensor) -> Tuple[str, Optional[int], torch.dtype]:
    device = tensor.device
    return (device.type, device.index if device.type != "cpu" else None, tensor.dtype)


def _key_to_device(key: Tuple[str, Optional[int], torch.dtype]) -> torch.device:
    device_type, index, _ = key
    if device_type == "cpu":
        return torch.device("cpu")
    if index is None:
        return torch.device(device_type)
    return torch.device(device_type, index)


def _ensure_weight(
    tensor: torch.Tensor,
    base: torch.Tensor,
    name: str,
) -> torch.Tensor:
    key = _device_key(tensor)
    device = _key_to_device(key)
    cache = _FAST_WEIGHTS.setdefault(key, {})
    if name not in cache or cache[name].dtype != tensor.dtype:
        cache[name] = base.to(device=device, dtype=tensor.dtype).contiguous()
    return cache[name]


def _spectral_to_xyz_fast(cube: torch.Tensor) -> torch.Tensor:
    if cube.dim() not in (3, 4):
        raise ValueError(f"Expected tensor of shape (B,C,H,W) or (C,H,W), got {cube.shape}")
    added_batch = False
    if cube.dim() == 3:
        cube = cube.unsqueeze(0)
        added_batch = True
    if cube.shape[1] != _XYZ_WEIGHTS.shape[0]:
        result = spectral_to_xyz(cube)
    else:
        weight = _ensure_weight(cube, _XYZ_CONV_WEIGHT_BASE, "xyz_conv_weight")
        result = F.conv2d(cube, weight)
    if added_batch:
        result = result.squeeze(0)
    return result


def _xyz_to_linear_srgb_fast(xyz: torch.Tensor) -> torch.Tensor:
    if xyz.dim() not in (3, 4):
        raise ValueError(f"Expected XYZ tensor of shape (B,3,H,W) or (3,H,W), got {xyz.shape}")
    added_batch = False
    if xyz.dim() == 3:
        xyz = xyz.unsqueeze(0)
        added_batch = True
    if xyz.shape[1] != 3:
        raise ValueError(f"XYZ tensor must have 3 channels, got {xyz.shape[1]}")
    weight = _ensure_weight(xyz, _SRGB_CONV_WEIGHT_BASE, "srgb_conv_weight")
    result = F.conv2d(xyz, weight)
    if added_batch:
        result = result.squeeze(0)
    return result


def _spectral_to_srgb_fast(cube: torch.Tensor, apply_gamma: bool = True) -> torch.Tensor:
    xyz = _spectral_to_xyz_fast(cube)
    linear = torch.clamp(_xyz_to_linear_srgb_fast(xyz), min=0.0)
    if apply_gamma:
        srgb = linear_to_srgb(linear)
    else:
        srgb = linear
    return torch.clamp(srgb, min=0.0, max=1.0)


def _spectral_to_lab_fast(cube: torch.Tensor) -> torch.Tensor:
    xyz = _spectral_to_xyz_fast(cube)
    return xyz_to_lab(xyz)


def _flatten_spectra(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected tensor of shape (B,C,H,W), got {x.shape}")
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


def _get_cached(
    cache: MutableMapping[str, torch.Tensor],
    key: str,
    factory: Callable[[], torch.Tensor],
) -> torch.Tensor:
    if key not in cache:
        cache[key] = factory()
    return cache[key]


def mae(pred: torch.Tensor, target: torch.Tensor, _: Optional[MutableMapping[str, torch.Tensor]] = None) -> float:
    return float(F.l1_loss(pred, target).item())


def mse(pred: torch.Tensor, target: torch.Tensor, _: Optional[MutableMapping[str, torch.Tensor]] = None) -> float:
    return float(F.mse_loss(pred, target).item())


def psnr_hsi(
    pred: torch.Tensor,
    target: torch.Tensor,
    _: Optional[MutableMapping[str, torch.Tensor]] = None,
    data_range: float = 1.0,
) -> float:
    err = F.mse_loss(pred, target)
    if err.item() == 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(data_range**2, device=err.device, dtype=err.dtype) / err).item())


def sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    _: Optional[MutableMapping[str, torch.Tensor]] = None,
    eps: float = 1e-8,
) -> float:
    """
    Spectral Angle Mapper in degrees, averaged over spatial locations.
    """
    p = _flatten_spectra(pred) + eps
    t = _flatten_spectra(target) + eps
    dot = torch.sum(p * t, dim=1)
    p_norm = torch.norm(p, dim=1)
    t_norm = torch.norm(t, dim=1)
    cos = torch.clamp(dot / (p_norm * t_norm + eps), -1.0, 1.0)
    ang = torch.arccos(cos)
    return float(torch.mean(torch.rad2deg(ang)).item())


def sid(
    pred: torch.Tensor,
    target: torch.Tensor,
    _: Optional[MutableMapping[str, torch.Tensor]] = None,
    eps: float = 1e-8,
) -> float:
    p = torch.clamp(_flatten_spectra(pred), min=eps)
    t = torch.clamp(_flatten_spectra(target), min=eps)
    p_norm = p / torch.sum(p, dim=1, keepdim=True)
    t_norm = t / torch.sum(t, dim=1, keepdim=True)
    divergence = torch.sum(p_norm * (torch.log(p_norm) - torch.log(t_norm)), dim=1)
    divergence += torch.sum(t_norm * (torch.log(t_norm) - torch.log(p_norm)), dim=1)
    return float(torch.mean(divergence).item())


def ergas(
    pred: torch.Tensor,
    target: torch.Tensor,
    _: Optional[MutableMapping[str, torch.Tensor]] = None,
    scale: float = 1.0,
    eps: float = 1e-8,
) -> float:
    diff = pred - target
    rmse = torch.sqrt(torch.mean(diff * diff, dim=(0, 2, 3)))
    mean_target = torch.mean(target, dim=(0, 2, 3))
    ratio = rmse / torch.clamp(mean_target, min=eps)
    value = 100.0 / scale * torch.sqrt(torch.mean(ratio * ratio))
    return float(value.item())


def psnr_srgb(
    pred: torch.Tensor,
    target: torch.Tensor,
    cache: Optional[MutableMapping[str, torch.Tensor]] = None,
    data_range: float = 1.0,
) -> float:
    if cache is None:
        cache = {}
    srgb_pred = _get_cached(cache, "srgb_gamma_pred", lambda: spectral_to_srgb(pred, apply_gamma=True))
    srgb_target = _get_cached(cache, "srgb_gamma_target", lambda: spectral_to_srgb(target, apply_gamma=True))
    err = F.mse_loss(srgb_pred, srgb_target)
    if err.item() == 0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(data_range**2, device=err.device, dtype=err.dtype) / err).item())


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.matmul(kernel_1d.unsqueeze(1), kernel_1d.unsqueeze(0))
    window = kernel_2d.unsqueeze(0).unsqueeze(0)
    return window.repeat(channels, 1, 1, 1)


def _ssim(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if pred_rgb.shape != target_rgb.shape:
        raise ValueError(f"Shape mismatch for SSIM: {pred_rgb.shape} vs {target_rgb.shape}")
    if pred_rgb.dim() != 4:
        raise ValueError("SSIM expects tensors of shape (B,3,H,W)")
    b, c, h, w = pred_rgb.shape
    if h < window_size or w < window_size:
        raise ValueError(f"SSIM window ({window_size}) larger than image size ({h}x{w})")
    pad = window_size // 2
    device = pred_rgb.device
    dtype = pred_rgb.dtype
    window = _gaussian_window(window_size, sigma, c, device, dtype)
    padded_pred = F.pad(pred_rgb, (pad, pad, pad, pad), mode="reflect")
    padded_target = F.pad(target_rgb, (pad, pad, pad, pad), mode="reflect")

    mu1 = F.conv2d(padded_pred, window, groups=c)
    mu2 = F.conv2d(padded_target, window, groups=c)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(padded_pred * padded_pred, window, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(padded_target * padded_target, window, groups=c) - mu2_sq
    sigma12 = F.conv2d(padded_pred * padded_target, window, groups=c) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator
    return ssim_map.mean(dim=(1, 2, 3))


def ssim_srgb(
    pred: torch.Tensor,
    target: torch.Tensor,
    cache: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> float:
    if cache is None:
        cache = {}
    srgb_pred = _get_cached(cache, "srgb_gamma_pred", lambda: spectral_to_srgb(pred, apply_gamma=True))
    srgb_target = _get_cached(cache, "srgb_gamma_target", lambda: spectral_to_srgb(target, apply_gamma=True))
    values = _ssim(srgb_pred, srgb_target)
    return float(values.mean().item())


def _delta_e_cie2000(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    if lab1.shape != lab2.shape:
        raise ValueError(f"Shape mismatch for ΔE00: {lab1.shape} vs {lab2.shape}")
    if lab1.dim() != 4:
        raise ValueError("ΔE00 expects tensors of shape (B,3,H,W)")
    b, _, h, w = lab1.shape
    flat1 = lab1.permute(0, 2, 3, 1).reshape(-1, 3)
    flat2 = lab2.permute(0, 2, 3, 1).reshape(-1, 3)

    L1, a1, b1 = flat1[:, 0], flat1[:, 1], flat1[:, 2]
    L2, a2, b2 = flat2[:, 0], flat2[:, 1], flat2[:, 2]

    C1 = torch.sqrt(a1 * a1 + b1 * b1)
    C2 = torch.sqrt(a2 * a2 + b2 * b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar.pow(7)
    G = 0.5 * (1 - torch.sqrt(C_bar7 / (C_bar7 + (25.0**7))))

    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    C1_prime = torch.sqrt(a1_prime * a1_prime + b1 * b1)
    C2_prime = torch.sqrt(a2_prime * a2_prime + b2 * b2)

    h1_prime = torch.atan2(b1, a1_prime)
    h2_prime = torch.atan2(b2, a2_prime)
    h1_prime = torch.remainder(h1_prime, 2 * torch.pi)
    h2_prime = torch.remainder(h2_prime, 2 * torch.pi)

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    delta_h_prime = h2_prime - h1_prime
    zero_mask = (C1_prime * C2_prime) == 0
    delta_h_prime = torch.where(
        zero_mask,
        torch.zeros_like(delta_h_prime),
        torch.where(
            delta_h_prime > torch.pi,
            delta_h_prime - 2 * torch.pi,
            torch.where(delta_h_prime < -torch.pi, delta_h_prime + 2 * torch.pi, delta_h_prime),
        ),
    )
    delta_H_prime = 2 * torch.sqrt(C1_prime * C2_prime) * torch.sin(delta_h_prime / 2.0)

    L_bar_prime = 0.5 * (L1 + L2)
    C_bar_prime = 0.5 * (C1_prime + C2_prime)

    h_bar_prime = torch.where(
        zero_mask,
        h1_prime + h2_prime,
        torch.where(
            torch.abs(h1_prime - h2_prime) > torch.pi,
            torch.where(h1_prime + h2_prime < 2 * torch.pi, (h1_prime + h2_prime + 2 * torch.pi) / 2.0, (h1_prime + h2_prime - 2 * torch.pi) / 2.0),
            (h1_prime + h2_prime) / 2.0,
        ),
    )

    T = (
        1
        - 0.17 * torch.cos(h_bar_prime - torch.deg2rad(torch.tensor(30.0, device=h_bar_prime.device, dtype=h_bar_prime.dtype)))
        + 0.24 * torch.cos(2 * h_bar_prime)
        + 0.32 * torch.cos(3 * h_bar_prime + torch.deg2rad(torch.tensor(6.0, device=h_bar_prime.device, dtype=h_bar_prime.dtype)))
        - 0.20 * torch.cos(4 * h_bar_prime - torch.deg2rad(torch.tensor(63.0, device=h_bar_prime.device, dtype=h_bar_prime.dtype)))
    )

    delta_theta = 30.0 * torch.exp(-(((torch.rad2deg(h_bar_prime) - 275.0) / 25.0) ** 2))
    R_C = 2 * torch.sqrt(C_bar_prime.pow(7) / (C_bar_prime.pow(7) + (25.0**7)))
    S_L = 1 + (0.015 * ((L_bar_prime - 50.0) ** 2)) / torch.sqrt(20.0 + ((L_bar_prime - 50.0) ** 2))
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T
    R_T = -torch.sin(2 * torch.deg2rad(delta_theta)) * R_C

    term_L = delta_L_prime / (S_L + _EPS)
    term_C = delta_C_prime / (S_C + _EPS)
    term_H = delta_H_prime / (S_H + _EPS)

    delta_E = torch.sqrt(
        term_L * term_L
        + term_C * term_C
        + term_H * term_H
        + R_T * term_C * term_H
    )
    return delta_E.view(b, h, w)


def deltae00(
    pred: torch.Tensor,
    target: torch.Tensor,
    cache: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> float:
    if cache is None:
        cache = {}
    lab_pred = _get_cached(cache, "lab_pred", lambda: spectral_to_lab(pred))
    lab_target = _get_cached(cache, "lab_target", lambda: spectral_to_lab(target))
    delta = _delta_e_cie2000(lab_pred, lab_target)
    return float(delta.mean().item())


def _flatten_hw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected tensor of shape (B,C,H,W), got {x.shape}")
    return x.flatten(start_dim=2)  # (B,C,HW)


def sam_fast(
    pred: torch.Tensor,
    target: torch.Tensor,
    _: Optional[MutableMapping[str, torch.Tensor]] = None,
    eps: float = 1e-8,
) -> float:
    spectra_pred = _flatten_hw(pred)
    spectra_target = _flatten_hw(target)
    dot = torch.sum(spectra_pred * spectra_target, dim=1)
    pred_norm = torch.norm(spectra_pred, dim=1)
    target_norm = torch.norm(spectra_target, dim=1)
    denom = pred_norm * target_norm + eps
    cos = torch.clamp(dot / denom, -1.0, 1.0)
    angles = torch.acos(cos)
    return float(torch.rad2deg(angles).mean().item())


def sid_fast(
    pred: torch.Tensor,
    target: torch.Tensor,
    _: Optional[MutableMapping[str, torch.Tensor]] = None,
    eps: float = 1e-8,
) -> float:
    spectra_pred = torch.clamp(_flatten_hw(pred), min=eps)
    spectra_target = torch.clamp(_flatten_hw(target), min=eps)
    pred_norm = spectra_pred / torch.sum(spectra_pred, dim=1, keepdim=True)
    target_norm = spectra_target / torch.sum(spectra_target, dim=1, keepdim=True)
    ratio = torch.clamp(pred_norm / target_norm, min=eps)
    log_ratio = torch.log(ratio)
    divergence = torch.sum((pred_norm - target_norm) * log_ratio, dim=1)
    return float(divergence.mean().item())


def ergas_fast(
    pred: torch.Tensor,
    target: torch.Tensor,
    _: Optional[MutableMapping[str, torch.Tensor]] = None,
    scale: float = 1.0,
    eps: float = 1e-8,
) -> float:
    diff = pred - target
    rmse = torch.sqrt(torch.mean(diff * diff, dim=(0, 2, 3)))
    mean_target = torch.mean(target, dim=(0, 2, 3))
    ratio = rmse / torch.clamp(mean_target, min=eps)
    value = 100.0 / scale * torch.sqrt(torch.mean(ratio * ratio))
    return float(value.item())


def psnr_srgb_fast(
    pred: torch.Tensor,
    target: torch.Tensor,
    cache: Optional[MutableMapping[str, torch.Tensor]] = None,
    data_range: float = 1.0,
) -> float:
    if cache is None:
        cache = {}
    srgb_pred = _get_cached(
        cache,
        "srgb_gamma_pred_fast",
        lambda: _spectral_to_srgb_fast(pred, apply_gamma=True),
    )
    srgb_target = _get_cached(
        cache,
        "srgb_gamma_target_fast",
        lambda: _spectral_to_srgb_fast(target, apply_gamma=True),
    )
    err = F.mse_loss(srgb_pred, srgb_target)
    if err.item() == 0:
        return float("inf")
    return float(
        10.0
        * torch.log10(
            torch.tensor(
                data_range**2,
                device=err.device,
                dtype=err.dtype,
            )
            / err
        ).item()
    )


def ssim_srgb_fast(
    pred: torch.Tensor,
    target: torch.Tensor,
    cache: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> float:
    if cache is None:
        cache = {}
    srgb_pred = _get_cached(
        cache,
        "srgb_gamma_pred_fast",
        lambda: _spectral_to_srgb_fast(pred, apply_gamma=True),
    )
    srgb_target = _get_cached(
        cache,
        "srgb_gamma_target_fast",
        lambda: _spectral_to_srgb_fast(target, apply_gamma=True),
    )
    values = _ssim(srgb_pred, srgb_target)
    return float(values.mean().item())


def deltae00_fast(
    pred: torch.Tensor,
    target: torch.Tensor,
    cache: Optional[MutableMapping[str, torch.Tensor]] = None,
) -> float:
    if cache is None:
        cache = {}
    lab_pred = _get_cached(cache, "lab_pred_fast", lambda: _spectral_to_lab_fast(pred))
    lab_target = _get_cached(cache, "lab_target_fast", lambda: _spectral_to_lab_fast(target))
    delta = _delta_e_cie2000(lab_pred, lab_target)
    return float(delta.mean().item())


psnr = psnr_hsi  # Backwards compatibility alias

_METRIC_REGISTRY: Dict[str, MetricFn] = {
    "mae": mae,
    "mse": mse,
    "psnr": psnr_hsi,
    "psnr_hsi": psnr_hsi,
    "psnr_srgb": psnr_srgb,
    "sam": sam,
    "sid": sid,
    "ergas": ergas,
    "ssim_srgb": ssim_srgb,
    "deltae00": deltae00,
}

_OPTIMIZED_METRICS: Dict[str, MetricFn] = {
    "sam_fast": sam_fast,
    "sid_fast": sid_fast,
    "ergas_fast": ergas_fast,
    "psnr_srgb_fast": psnr_srgb_fast,
    "ssim_srgb_fast": ssim_srgb_fast,
    "deltae00_fast": deltae00_fast,
}


def list_available_metrics() -> List[str]:
    return sorted(_METRIC_REGISTRY.keys())


def aggregate(
    pred: torch.Tensor,
    target: torch.Tensor,
    metrics: Iterable[str] | None = None,
    *,
    include_timings: bool = False,
) -> Dict[str, float]:
    """
    Compute the selected metrics and return them as a dictionary.
    """
    metric_names = list(metrics) if metrics is not None else list_available_metrics()
    cache: Dict[str, torch.Tensor] = {}
    results: Dict[str, float] = {}
    timings: Dict[str, float] = {}
    for name in metric_names:
        if name not in _METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{name}'. Available: {list_available_metrics()}")
        start = time.perf_counter() if include_timings else None
        results[name] = _METRIC_REGISTRY[name](pred, target, cache)
        if include_timings and start is not None:
            timings[name + "_ms"] = (time.perf_counter() - start) * 1000.0
    if include_timings:
        results["_timings"] = timings
    return results


__all__ = [
    "mae",
    "mse",
    "psnr",
    "psnr_hsi",
    "psnr_srgb",
    "sam",
    "sid",
    "ergas",
    "ssim_srgb",
    "deltae00",
    "aggregate",
    "list_available_metrics",
    "sam_fast",
    "sid_fast",
    "ergas_fast",
    "psnr_srgb_fast",
    "ssim_srgb_fast",
    "deltae00_fast",
    "_OPTIMIZED_METRICS",
]

