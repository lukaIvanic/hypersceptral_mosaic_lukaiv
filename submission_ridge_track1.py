import argparse
import os
from pathlib import Path
import numpy as np
import zipfile
from torch.utils.data import DataLoader
from datasets.hyper_object import HyperObjectDataset
from classical.featurize import pack_2x2, build_features_from_packed, unpack_pred_to_cube


def load_ridge_model(path: str):
    d = np.load(path)
    return d["coefs"], d["inter"]  # (Cout,D), (Cout,1)


def predict_packed(coefs: np.ndarray, inter: np.ndarray, X: np.ndarray) -> np.ndarray:
    # X: (N,D)
    # coefs: (Cout,D)
    Y = X @ coefs.T + inter.squeeze(1)
    return Y


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    ds = HyperObjectDataset(data_root=args.data_dir, track=1, train=False, submisison=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    coefs, inter = load_ridge_model(args.model)

    ids = []
    for batch in loader:
        sid = batch["id"][0]
        mosaic = batch["input"][0].numpy()      # (1,H,W)
        packed = pack_2x2(mosaic)                 # (4,H2,W2)
        X, (H2, W2) = build_features_from_packed(packed)
        Y = predict_packed(coefs, inter, X)       # (N, Cout)
        Cout = coefs.shape[0]
        Yhw = Y.T.reshape(Cout, H2, W2)
        cube = unpack_pred_to_cube(Yhw, bands=61) # (61,H,W)
        cube = np.clip(cube, 0.0, 1.0)
        cube_hwc = np.transpose(cube, (1, 2, 0))
        np.savez_compressed(os.path.join(args.out_dir, f"{sid}.npz"), cube=cube_hwc)
        ids.append(sid)

    # Write CSV + Zip
    csv_path = os.path.join(args.out_dir, "submission.csv")
    import pandas as pd
    pd.DataFrame({"id": ids, "prediction": 0}).to_csv(csv_path, index=False)
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.csv")
        for sid in ids:
            zf.write(os.path.join(args.out_dir, f"{sid}.npz"), arcname=f"{sid}.npz")
    print(f"[OK] Wrote {args.zip_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/track1")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="submission_files_track1_ridge")
    p.add_argument("--zip_path", type=str, default="submission_track1_ridge.zip")
    args = p.parse_args()
    main(args)

