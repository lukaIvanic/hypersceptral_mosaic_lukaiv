#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


def load_gt_cube(gt_root: Path, sample_id: str) -> np.ndarray:
    h5_path = gt_root / f"{sample_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"GT cube missing: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        cube = np.array(f["cube"], dtype=np.float32)
    return cube  # (H, W, 61)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check exported submission cubes.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Folder with <id>.npz files.")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="How many cubes to inspect (0 = all).",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Optional directory with ground-truth .h5 files (for val/public test).",
    )
    args = parser.parse_args()

    pred_dir = args.pred_dir.expanduser().resolve()
    files = sorted(pred_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in {pred_dir}")

    if args.sample_limit > 0:
        files = files[: args.sample_limit]

    for path in files:
        sample_id = path.stem
        data = np.load(path)
        if "cube" not in data:
            raise KeyError(f"{path.name} lacks 'cube' array")
        cube = data["cube"]  # expected (H, W, 61)

        finite = np.isfinite(cube).all()
        min_val = float(cube.min())
        max_val = float(cube.max())
        shape = cube.shape

        print(f"{path.name}: shape={shape}, finite={finite}, min={min_val:.4f}, max={max_val:.4f}")

        if shape[-1] != 61:
            print(f"  !! Expected last dimension 61, got {shape[-1]}")
        if min_val < 0 or max_val > 1:
            print("  !! Values outside [0,1]")

        if args.gt_dir is not None:
            if h5py is None:
                raise RuntimeError("h5py not installed; install it to compare with ground truth.")
            gt_cube = load_gt_cube(args.gt_dir, sample_id)
            if gt_cube.shape != cube.shape:
                print(f"  GT mismatch: {gt_cube.shape=} vs {cube.shape=}")
            diff = np.abs(cube - gt_cube)
            print(
                f"  GT diff stats -> mean={diff.mean():.5f}, max={diff.max():.5f}, "
                f"SAM placeholder (compute separately if needed)."
            )


if __name__ == "__main__":
    main()