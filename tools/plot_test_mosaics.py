from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required to run this script. Install it with 'pip install matplotlib'.") from exc

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is required to run this script. Install it with 'pip install torch'.") from exc


_DEMOSAIC_KERNEL = torch.tensor(
    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32
).view(1, 1, 3, 3)


def mosaic_to_rgb(mosaic: np.ndarray) -> np.ndarray:
    """
    Convert an RGGB Bayer mosaic (values in [0, 1]) to an RGB image using
    simple bilinear demosaicing.
    """
    array = np.asarray(mosaic)
    if array.ndim == 3:
        # Allow singleton channel dimensions such as (H, W, 1) or (1, H, W)
        squeezed = np.squeeze(array)
        if squeezed.ndim != 2:
            raise ValueError(f"Expected mosaic with 1 channel, got shape {array.shape}")
        array = squeezed
    if array.ndim != 2:
        raise ValueError(f"Expected 2D mosaic array, got shape {array.shape}")

    h, w = array.shape
    if h < 2 or w < 2:
        raise ValueError(f"Mosaic is too small for demosaicing: shape {array.shape}")

    m = torch.from_numpy(array.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    kernel = _DEMOSAIC_KERNEL.to(device=m.device, dtype=m.dtype)

    mask_r = torch.zeros_like(m)
    mask_g = torch.zeros_like(m)
    mask_b = torch.zeros_like(m)

    mask_r[:, :, 0::2, 0::2] = 1.0  # R
    mask_g[:, :, 0::2, 1::2] = 1.0  # G (even rows)
    mask_g[:, :, 1::2, 0::2] = 1.0  # G (odd rows)
    mask_b[:, :, 1::2, 1::2] = 1.0  # B

    def _interp(mask: torch.Tensor) -> torch.Tensor:
        values = m * mask
        num = F.conv2d(values, kernel, padding=1)
        den = F.conv2d(mask, kernel, padding=1)
        channel = torch.zeros_like(num)
        valid = den > 0
        channel[valid] = num[valid] / den[valid]
        return channel

    r = _interp(mask_r)
    g = _interp(mask_g)
    b = _interp(mask_b)
    rgb = torch.cat([r, g, b], dim=1).squeeze(0).permute(1, 2, 0)
    return torch.clamp(rgb, 0.0, 1.0).detach().cpu().numpy()


def _glob_mosaic_paths(mosaic_dir: Path, limit: int) -> List[Path]:
    paths = sorted(mosaic_dir.glob("*.npy"))
    if limit > 0:
        paths = paths[:limit]
    return paths


def _compute_layout(total: int, cols: int) -> Tuple[int, int]:
    cols = max(1, cols)
    rows = max(1, math.ceil(total / cols))
    return rows, cols


def plot_mosaics(
    paths: Sequence[Path],
    cols: int,
    figsize: Tuple[float, float],
    save_path: Path | None,
    show: bool,
) -> None:
    if not paths:
        print("[WARN] No .npy mosaics found to plot.")
        return

    rows, cols = _compute_layout(len(paths), cols)
    # Compute overall figure size by scaling per-subplot size.
    fig_w = figsize[0] * cols
    fig_h = figsize[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    # Normalise axes handling for grids of 1 row/column.
    if isinstance(axes, plt.Axes):
        axes_iter: Iterable[plt.Axes] = [axes]
    else:
        axes_iter = axes.flatten()

    axes_list = list(axes_iter)
    for idx, path in enumerate(paths):
        ax = axes_list[idx]
        mosaic = np.load(path)
        rgb = mosaic_to_rgb(mosaic)

        ax.imshow(rgb)
        ax.set_title(path.stem, fontsize=9)
        ax.axis("off")

    # Hide any unused subplot axes.
    for ax in axes_list[len(paths) :]:
        ax.axis("off")

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved mosaic grid -> {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def _default_mosaic_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "data" / "track1" / "test_original" / "mosaic"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RGB previews for Track 1 test mosaics stored as .npy files."
    )
    parser.add_argument(
        "--mosaic-dir",
        type=Path,
        default=None,
        help="Directory containing mosaic .npy files (default: data/track1/test_original/mosaic).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of mosaics plotted (0 means all found).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of columns in the subplot grid (default: 3).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(4.0, 4.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Size of each subplot in inches (default: 4 4).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the resulting figure (e.g., outputs/test_mosaic_grid.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the figure interactively.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    mosaic_dir = args.mosaic_dir or _default_mosaic_dir()
    mosaic_dir = mosaic_dir.expanduser().resolve()

    if not mosaic_dir.exists():
        raise FileNotFoundError(f"Mosaic directory not found: {mosaic_dir}")

    paths = _glob_mosaic_paths(mosaic_dir, args.limit)
    if not paths:
        print(f"[WARN] No .npy files found under {mosaic_dir}")
        return

    save_path = args.save.expanduser().resolve() if args.save else None
    plot_mosaics(
        paths=paths,
        cols=args.cols,
        figsize=tuple(args.figsize),
        save_path=save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()


