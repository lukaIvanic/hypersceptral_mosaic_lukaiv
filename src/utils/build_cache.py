from __future__ import annotations

import argparse
from pathlib import Path

from src.data import Track1Dataset


def build_cache(
    data_root: Path,
    split: str,
    resize_to: int,
    cache_dir: Path,
) -> None:
    dataset = Track1Dataset(
        root=data_root,
        split=split,
        augment=False,
        resize_to=resize_to,
        cache_dir=cache_dir,
        write_cache=True,
    )
    total = len(dataset)
    print(f"[Cache] Building cache for split='{split}' size={resize_to} ({total} samples)")
    for idx in range(total):
        dataset[idx]  # triggers cache write
        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(f"[Cache] {idx + 1}/{total} complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute resized cache for Track 1 data")
    parser.add_argument("--data-root", type=str, default="data/track1", help="Path to Track 1 data root")
    parser.add_argument("--cache-dir", type=str, default="data/cache/track1", help="Destination cache directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "validation", "test_public", "test"], help="Dataset split to cache")
    parser.add_argument("--size", type=int, default=64, help="Spatial size to resize to (NxN)")
    args = parser.parse_args()

    split = args.split
    if split == "validation":
        split = "val"
    elif split == "test_public":
        split = "val"

    build_cache(
        data_root=Path(args.data_root),
        split=split,
        resize_to=args.size,
        cache_dir=Path(args.cache_dir),
    )


if __name__ == "__main__":
    main()


