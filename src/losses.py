from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .color import spectral_to_srgb


def _symmetric_pad(window_size: int) -> int:
    if window_size % 2 == 0:
        raise ValueError("SSIM window size must be odd.")
    return window_size // 2


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.matmul(kernel_1d.unsqueeze(1), kernel_1d.unsqueeze(0))
    window = kernel_2d.expand(channels, 1, window_size, window_size)
    return window


def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {pred.shape} vs {target.shape}")
    if pred.dim() != 4:
        raise ValueError("SSIM expects tensors of shape (B,3,H,W)")
    b, c, h, w = pred.shape
    if h < window_size or w < window_size:
        raise ValueError(
            f"SSIM window ({window_size}) larger than image size ({h}x{w})"
        )
    pad = _symmetric_pad(window_size)
    device = pred.device
    dtype = pred.dtype
    window = _gaussian_window(window_size, sigma, c, device, dtype)

    padded_pred = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
    padded_target = F.pad(target, (pad, pad, pad, pad), mode="reflect")

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


def _flatten_spectra(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W) tensor, got {tensor.shape}")
    return tensor.permute(0, 2, 3, 1).reshape(-1, tensor.shape[1])


def spectral_angle_mapper_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    use_angle: bool = False,
    return_degrees: bool = False,
) -> torch.Tensor:
    """
    Differentiable SAM-style loss averaged over spatial locations.

    When ``use_angle`` is False (default), returns 1 - cosine similarity, which
    is smoother near zero and avoids arccos gradients exploding when spectra
    align closely. Set ``use_angle=True`` to recover the true angular error.
    """
    p = _flatten_spectra(pred)
    t = _flatten_spectra(target)
    dot = torch.sum(p * t, dim=1)
    p_norm = torch.linalg.norm(p, ord=2, dim=1).clamp(min=eps)
    t_norm = torch.linalg.norm(t, ord=2, dim=1).clamp(min=eps)

    cosine = torch.clamp(dot / (p_norm * t_norm), -1.0 + 1e-4, 1.0 - 1e-4)

    if not use_angle:
        return torch.mean(1.0 - cosine)

    angles = torch.arccos(cosine)
    if return_degrees:
        angles = torch.rad2deg(angles)
    return torch.mean(angles)


def ergas_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: float = 1.0,
    eps: float = 1e-6,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Differentiable ERGAS loss matching the evaluation metric.

    By default the value is normalized (≈0–1 range) to keep gradient magnitudes
    comparable to the other reconstruction terms. Disable normalization if you
    need the raw ERGAS figure.
    """
    diff = pred - target
    rmse = torch.sqrt(torch.mean(diff * diff, dim=(0, 2, 3)) + eps)
    mean_target = torch.mean(target, dim=(0, 2, 3)).clamp(min=eps)
    ratio = rmse / mean_target
    value = 100.0 / scale * torch.sqrt(torch.mean(ratio * ratio) + eps)
    if normalize:
        return value / 100.0
    return value


@dataclass
class LossWeights:
    lambda_l1: float = 1.0
    lambda_sam: float = 0.0
    lambda_sid: float = 0.0
    lambda_srgb_l1: float = 0.0
    lambda_srgb_ssim: float = 0.0
    lambda_ergas: float = 0.0


class CompositeSpectralLoss(nn.Module):
    """
    Combine reconstruction losses with configurable weights.
    Currently supports L1 and SAM; additional terms can be added as needed.
    """

    def __init__(self, weights: LossWeights | None = None, eps: float = 1e-8) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        terms: Dict[str, torch.Tensor] = {}
        if self.weights.lambda_l1:
            terms["l1"] = F.l1_loss(pred, target)
        if self.weights.lambda_sam:
            terms["sam"] = spectral_angle_mapper_loss(
                pred, target, eps=self.eps, use_angle=False
            )
        if self.weights.lambda_sid:
            terms["sid"] = spectral_information_divergence_loss(
                pred, target, eps=self.eps
            )
        if self.weights.lambda_ergas:
            terms["ergas"] = ergas_loss(pred, target, eps=self.eps)
        need_srgb = self.weights.lambda_srgb_l1 or self.weights.lambda_srgb_ssim
        if need_srgb:
            srgb_linear_pred = spectral_to_srgb(pred, apply_gamma=False)
            srgb_linear_target = spectral_to_srgb(target, apply_gamma=False)
            srgb_gamma_pred = spectral_to_srgb(pred, apply_gamma=True)
            srgb_gamma_target = spectral_to_srgb(target, apply_gamma=True)
            terms["srgb_l1"] = F.l1_loss(
                srgb_linear_pred, srgb_linear_target
            )
            if self.weights.lambda_srgb_ssim:
                ssim_vals = _ssim(
                    srgb_gamma_pred,
                    srgb_gamma_target,
                    data_range=1.0,
                    window_size=11,
                    sigma=1.5,
                )
                terms["srgb_ssim"] = 1.0 - ssim_vals.mean()

        if not terms:
            raise ValueError("CompositeSpectralLoss requires at least one active term.")

        loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        if "l1" in terms:
            loss = loss + self.weights.lambda_l1 * terms["l1"]
        if "sam" in terms:
            loss = loss + self.weights.lambda_sam * terms["sam"]
        if "sid" in terms:
            loss = loss + self.weights.lambda_sid * terms["sid"]
        if "ergas" in terms:
            loss = loss + self.weights.lambda_ergas * terms["ergas"]
        if "srgb_l1" in terms:
            loss = loss + self.weights.lambda_srgb_l1 * terms["srgb_l1"]
        if "srgb_ssim" in terms:
            loss = loss + self.weights.lambda_srgb_ssim * terms["srgb_ssim"]
        return loss


def spectral_information_divergence_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Symmetric SID loss averaged over spatial locations.
    """
    p = _flatten_spectra(pred)
    t = _flatten_spectra(target)

    p = torch.clamp(p, min=eps)
    t = torch.clamp(t, min=eps)

    p_norm = p / torch.sum(p, dim=1, keepdim=True).clamp(min=eps * p.shape[1])
    t_norm = t / torch.sum(t, dim=1, keepdim=True).clamp(min=eps * t.shape[1])

    log_ratio_pt = torch.log(p_norm) - torch.log(t_norm)
    log_ratio_tp = -log_ratio_pt

    divergence = torch.sum(p_norm * log_ratio_pt, dim=1) + torch.sum(
        t_norm * log_ratio_tp, dim=1
    )
    return divergence.mean()


__all__ = [
    "CompositeSpectralLoss",
    "LossWeights",
    "spectral_angle_mapper_loss",
    "ergas_loss",
    "spectral_information_divergence_loss",
]

