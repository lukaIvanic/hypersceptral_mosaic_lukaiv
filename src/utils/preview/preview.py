from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..data import Track1Dataset


def mosaic_to_rgb(mosaic: np.ndarray) -> np.ndarray:
    """
    Simple RGGB demosaicing to pseudo-RGB using bilinear interpolation.
    """
    if mosaic.ndim == 3:
        mosaic = mosaic.squeeze()
    m = torch.from_numpy(mosaic).float().unsqueeze(0).unsqueeze(0)

    mask_r = torch.zeros_like(m)
    mask_g = torch.zeros_like(m)
    mask_b = torch.zeros_like(m)

    mask_r[:, :, 0::2, 0::2] = 1  # R
    mask_g[:, :, 0::2, 1::2] = 1  # G on even rows
    mask_g[:, :, 1::2, 0::2] = 1  # G on odd rows
    mask_b[:, :, 1::2, 1::2] = 1  # B

    kernel = torch.tensor(
        [[1.0, 2.0, 1.0],
         [2.0, 4.0, 2.0],
         [1.0, 2.0, 1.0]],
        dtype=torch.float32,
    ).view(1, 1, 3, 3)

    def _interp(mask: torch.Tensor) -> torch.Tensor:
        values = m * mask
        num = F.conv2d(values, kernel, padding=1)
        den = F.conv2d(mask, kernel, padding=1)
        channel = torch.zeros_like(num)
        channel[den > 0] = num[den > 0] / den[den > 0]
        return channel

    r = _interp(mask_r)
    g = _interp(mask_g)
    b = _interp(mask_b)
    rgb = torch.cat([r, g, b], dim=1).squeeze().permute(1, 2, 0).clamp(0, 1)
    return rgb.detach().cpu().numpy()


def create_rgb_from_cube(
    cube: torch.Tensor,
    bands: Tuple[int, int, int] = (10, 20, 30),
) -> np.ndarray:
    """
    Convert a hyperspectral cube (61, H, W) to a false-colour RGB image.

    Args:
        cube: Torch tensor in shape (C, H, W) with values in [0, 1].
        bands: Indices of the spectral bands to map to R, G, B.
    """
    c, h, w = cube.shape
    for b in bands:
        if b < 0 or b >= c:
            raise ValueError(f"Band index {b} out of range for cube with {c} bands.")
    rgb = cube[list(bands), :, :].detach().cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def extract_band(cube: torch.Tensor, band: int) -> np.ndarray:
    c, h, w = cube.shape
    if band < 0 or band >= c:
        raise ValueError(f"Band index {band} out of range for cube with {c} bands.")
    band_img = cube[band].detach().cpu().numpy()
    band_min, band_max = band_img.min(), band_img.max()
    if band_max > band_min:
        band_img = (band_img - band_min) / (band_max - band_min)
    return band_img


def main(args: argparse.Namespace) -> None:
    dataset = Track1Dataset(root=Path(args.data_root), split="train")
    index = max(0, min(args.index, len(dataset) - 1))
    sample = dataset[index]

    mosaic = sample["input"].squeeze(0).numpy()
    cube = sample["target"]

    print(f"[Info] Sample '{sample['id']}'")
    print(f"       mosaic shape: {sample['input'].shape}")
    print(f"       cube shape:   {cube.shape}")

    fig, axes = plt.subplots(1, 1 if args.input_only else 2, figsize=(6 if args.input_only else 10, 4))

    if args.input_only:
        ax_mosaic = axes
        if args.grayscale_input:
            ax_mosaic.imshow(mosaic, cmap="gray")
        else:
            ax_mosaic.imshow(mosaic_to_rgb(mosaic))
        ax_mosaic.set_title(f"Mosaic | {sample['id']}")
        ax_mosaic.axis("off")
    else:
        ax_mosaic, ax_other = axes
        if args.grayscale_input:
            ax_mosaic.imshow(mosaic, cmap="gray")
        else:
            ax_mosaic.imshow(mosaic_to_rgb(mosaic))
        ax_mosaic.set_title(f"Mosaic | {sample['id']}")
        ax_mosaic.axis("off")

        if args.band is not None:
            view = extract_band(cube, args.band)
            ax_other.imshow(view, cmap="viridis")
            ax_other.set_title(f"HSI band {args.band}")
        else:
            view = create_rgb_from_cube(cube, bands=tuple(args.bands))
            ax_other.imshow(view)
            ax_other.set_title(f"Pseudo-RGB bands {args.bands}")
        ax_other.axis("off")

    fig.tight_layout()
    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"[OK] Saved preview to {out_path}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview a sample mosaic + pseudo-RGB render from the Track 1 dataset."
    )
    parser.add_argument("--data-root", type=str, default="data/track1", help="Track 1 data directory")
    parser.add_argument("--index", type=int, default=0, help="Sample index to preview")
    parser.add_argument(
        "--bands",
        type=int,
        nargs=3,
        default=(30, 15, 0),
        metavar=("R", "G", "B"),
        help="Spectral band indices to use for pseudo-RGB (default 30,15,0 ~ R,G,B)",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=None,
        help="Show a single spectral band instead of pseudo-RGB",
    )
    parser.add_argument("--out-path", type=str, default=None, help="Optional path to save figure")
    parser.add_argument(
        "--input-only",
        action="store_true",
        help="Visualise only the input mosaic (skip hyperspectral view)",
    )
    parser.add_argument(
        "--grayscale-input",
        action="store_true",
        help="Display mosaic as grayscale instead of pseudo-RGB demosaic",
    )
    parser.add_argument("--no-show", action="store_true", help="Skip interactive window (useful on headless servers)")
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)


