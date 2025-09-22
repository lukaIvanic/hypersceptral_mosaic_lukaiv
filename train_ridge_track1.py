import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets.hyper_object import HyperObjectDataset

from classical.featurize import pack_2x2, build_features_from_packed, pack_cube_target
from classical.ridge_model import RidgePerChannel


def main(args):
    ds = HyperObjectDataset(data_root=args.data_dir, track=1, train=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    X_list = []
    Y_list = []
    rng = np.random.default_rng(args.seed)

    for batch in tqdm(loader, total=len(loader)):
        mosaic = batch["input"][0].numpy()   # (1,H,W)
        cube   = batch["output"][0].numpy()  # (61,H,W)
        packed_m = pack_2x2(mosaic)           # (4,H2,W2)
        X, hw = build_features_from_packed(packed_m)
        packed_y = pack_cube_target(cube)     # (61*4,H2,W2)
        Y = packed_y.reshape(packed_y.shape[0], -1).T
        # Random subsample pixels to limit memory
        n = X.shape[0]
        take = min(args.samples_per_img, n)
        idx = rng.choice(n, size=take, replace=False)
        X_list.append(X[idx])
        Y_list.append(Y[idx])

    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)

    model = RidgePerChannel(alpha=args.alpha)
    model.fit(X_all, Y_all)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / "ridge_model.npz"
    # Save coefficients and intercepts
    coefs = np.stack([m.coef_.astype(np.float32) for m in model.models], axis=0)
    inter = np.stack([np.array([m.intercept_], dtype=np.float32) for m in model.models], axis=0)
    np.savez_compressed(out_path, coefs=coefs, inter=inter)
    print(f"[OK] Saved ridge model to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/track1")
    p.add_argument("--out_dir", type=str, default="runs/track1/ridge")
    p.add_argument("--samples_per_img", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)

