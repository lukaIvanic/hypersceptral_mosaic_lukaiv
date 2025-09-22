import argparse
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import torch

from datasets.hyper_object import HyperObjectDataset
from torch.utils.data import DataLoader


def main(args):
    ds = HyperObjectDataset(
        data_root=args.data_dir,
        track=1,
        train=True,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    ipca = IncrementalPCA(n_components=args.k, batch_size=args.ipca_batch)

    # First pass: fit PCA on a subset of pixels per image for speed
    rng = np.random.default_rng(args.seed)
    for batch in tqdm(loader, total=len(loader)):
        cube = batch["output"][0].numpy()  # (C,H,W)
        C, H, W = cube.shape
        # subsample pixels
        n_pix = min(args.samples_per_img, H * W)
        rr = rng.integers(0, H, size=n_pix)
        cc = rng.integers(0, W, size=n_pix)
        X = cube[:, rr, cc].T  # (n_pix, C)
        ipca.partial_fit(X)

    mean = ipca.mean_.astype(np.float32)
    components = ipca.components_.astype(np.float32)  # (K,C)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pca_basis_k{args.k}.npz"
    np.savez_compressed(out_path, mean=mean, components=components)
    print(f"[OK] Saved PCA basis to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/track1")
    p.add_argument("--out_dir", type=str, default="runs/track1/pca")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--ipca_batch", type=int, default=16384)
    p.add_argument("--samples_per_img", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)

