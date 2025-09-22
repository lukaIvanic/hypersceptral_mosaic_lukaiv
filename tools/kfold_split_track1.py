import argparse
from pathlib import Path
import json
from collections import defaultdict

def extract_category(stem: str) -> str:
    # Example stem: Category-1_a_0008
    if stem.startswith("Category-"):
        return stem.split("_")[0]
    return "unknown"

def main(args):
    import os
    from datasets.hyper_object import HyperObjectDataset

    ds = HyperObjectDataset(data_root=args.data_dir, track=1, train=True)
    ids = list(ds.ids)

    # Group by category for stratification
    by_cat = defaultdict(list)
    for sid in ids:
        by_cat[extract_category(sid)].append(sid)

    folds = [list() for _ in range(args.k)]
    for cat, sids in by_cat.items():
        sids.sort()
        for i, sid in enumerate(sids):
            folds[i % args.k].append(sid)

    out = {f"fold_{i}": sorted(folds[i]) for i in range(args.k)}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"kfold_{args.k}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] Saved splits to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/track1")
    p.add_argument("--out_dir", type=str, default="runs/track1/splits")
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()
    main(args)

