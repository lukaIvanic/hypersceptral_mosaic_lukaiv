#!/usr/bin/env python3
"""
Rebalances the validation split for Track 1 by ensuring each category
is equally represented. The script combines the existing train and
test-public splits, then rebuilds the validation set so that exactly
four samples from each category remain in `test-public`, with the rest
residing in `train`. The `test_original` split is left untouched.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


CATEGORY_PATTERN = re.compile(r"Category-(\d)_")
MODALITIES = ("hsi_61", "mosaic")


def extract_category(sample_name: str) -> int:
    match = CATEGORY_PATTERN.match(sample_name)
    if not match:
        raise ValueError(f"Unexpected sample name format: {sample_name}")
    return int(match.group(1))


def collect_samples(root: Path) -> Set[str]:
    hsi_dir = root / "hsi_61"
    if not hsi_dir.is_dir():
        raise FileNotFoundError(f"Missing modality directory: {hsi_dir}")
    return {
        path.stem
        for path in hsi_dir.iterdir()
        if path.is_file()
    }


def resolve_file(modality_dir: Path, sample_name: str) -> Path:
    matches = list(modality_dir.glob(f"{sample_name}.*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file for sample {sample_name} in {modality_dir}, "
            f"found {len(matches)}."
        )
    return matches[0]


def select_validation_samples(
    all_samples: Iterable[str],
    current_val: Set[str],
    per_category: int = 4,
    shuffle: bool = True,
    prefer_existing: bool = False,
    rng: Optional[random.Random] = None,
) -> Set[str]:
    grouped: Dict[int, List[str]] = defaultdict(list)
    for sample in all_samples:
        grouped[extract_category(sample)].append(sample)

    new_val: Set[str] = set()
    for category in sorted(grouped):
        available = sorted(grouped[category])
        if shuffle:
            if rng is None:
                rng = random.Random()
            available = available.copy()
            rng.shuffle(available)

        chosen: List[str] = []
        if prefer_existing:
            for sample in available:
                if sample in current_val:
                    chosen.append(sample)
                    if len(chosen) == per_category:
                        break

        for sample in available:
            if len(chosen) == per_category:
                break
            if sample in chosen:
                continue
            chosen.append(sample)

        if len(chosen) < per_category:
            raise RuntimeError(
                f"Category {category} has only {len(chosen)} samples; "
                f"{per_category} needed for a balanced validation set."
            )

        new_val.update(chosen)

    return new_val


def move_sample(sample: str, source_root: Path, dest_root: Path) -> None:
    for modality in MODALITIES:
        src_dir = source_root / modality
        dst_dir = dest_root / modality
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Expected modality directory: {src_dir}")
        if not dst_dir.is_dir():
            raise FileNotFoundError(f"Expected modality directory: {dst_dir}")

        src_file = resolve_file(src_dir, sample)
        dst_file = dst_dir / src_file.name
        if dst_file.exists():
            raise FileExistsError(f"Destination file already exists: {dst_file}")
        shutil.move(str(src_file), str(dst_file))


def summarize_selection(selection: Set[str]) -> str:
    by_category: Dict[int, List[str]] = defaultdict(list)
    for sample in selection:
        by_category[extract_category(sample)].append(sample)

    lines = []
    for category in sorted(by_category):
        names = ", ".join(sorted(by_category[category]))
        lines.append(f"  Category {category}: {names}")
    return "\n".join(lines)


def main(
    *,
    dry_run: bool = False,
    shuffle: bool = True,
    seed: Optional[int] = None,
    force: bool = False,
) -> None:
    base_dir = Path(__file__).resolve().parent / "track1"
    train_root = base_dir / "train"
    val_root = base_dir / "test-public"

    if not train_root.exists():
        raise FileNotFoundError(f"Missing train directory: {train_root}")
    if not val_root.exists():
        raise FileNotFoundError(f"Missing validation directory: {val_root}")

    train_samples = collect_samples(train_root)
    val_samples = collect_samples(val_root)
    all_samples = sorted(train_samples | val_samples)

    if not all_samples:
        raise RuntimeError("No samples found in train or test-public directories.")

    rng = random.Random(seed) if seed is not None else None
    if seed is not None:
        shuffle = True

    new_val_samples = select_validation_samples(
        all_samples,
        val_samples,
        shuffle=shuffle,
        prefer_existing=not force,
        rng=rng,
    )
    new_train_samples = set(all_samples) - new_val_samples

    to_val = sorted(new_val_samples - val_samples)
    to_train = sorted(val_samples - new_val_samples)

    print("Planned balanced validation selection:")
    print(summarize_selection(new_val_samples))
    print()
    print(f"Current counts -> train: {len(train_samples)}, val: {len(val_samples)}")
    print(
        f"Adjusted counts -> train: {len(new_train_samples)}, val: {len(new_val_samples)}"
    )
    print()
    if to_train:
        print("Samples to move from validation to train:")
        for sample in to_train:
            print(f"  - {sample}")
    if to_val:
        print("Samples to move from train to validation:")
        for sample in to_val:
            print(f"  - {sample}")
    if not to_train and not to_val:
        message = "Validation set already balanced; no file moves required."
        if force and not shuffle:
            message += " Use --shuffle to pick a different combination."
        print(message)

    if dry_run:
        return
    if not force and not to_train and not to_val:
        return

    for sample in to_train:
        move_sample(sample, val_root, train_root)
    for sample in to_val:
        move_sample(sample, train_root, val_root)

    print("\nReordering complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorder train/test-public splits with balanced categories."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the planned moves without modifying any files.",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable randomization (pick first available samples).",
    )
    parser.set_defaults(shuffle=True)
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the shuffling RNG. Implies --shuffle.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-selection even if the current validation split is balanced.",
    )
    args = parser.parse_args()
    main(
        dry_run=args.dry_run,
        shuffle=args.shuffle,
        seed=args.seed,
        force=args.force,
    )

