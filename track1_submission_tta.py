import os
import zipfile
import argparse
import time
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from datasets.hyper_object import HyperObjectDataset
from baselines.raw2hsi import Raw2HSI
from baselines.linear_mapper import LinearRaw2HSI
from baselines.pca_mapper import PCABasis, PCACoeffMapper
from config.track1_cfg import TrainerCfg


def _now() -> float:
    return time.perf_counter()


def tta_transforms(x: torch.Tensor):
    # x: (1,1,H,W)
    outs = []
    invs = []
    outs.append(x); invs.append(lambda y: y)
    outs.append(torch.flip(x, dims=[-1])); invs.append(lambda y: torch.flip(y, dims=[-1]))
    outs.append(torch.flip(x, dims=[-2])); invs.append(lambda y: torch.flip(y, dims=[-2]))
    outs.append(torch.transpose(x, -1, -2)); invs.append(lambda y: torch.transpose(y, -1, -2))
    return outs, invs


def load_model(ckpt_path, device, model_type: str = "linear", pca_basis_path: str = None):
    cfg = TrainerCfg()
    if model_type == "linear":
        model = LinearRaw2HSI(out_bands=cfg.out_bands).to(device)
    elif model_type == "pca":
        if pca_basis_path is None:
            raise ValueError("--pca_basis is required for pca model")
        data = np.load(pca_basis_path)
        basis = PCABasis(mean=data["mean"], components=data["components"]).to_torch(device)
        model = PCACoeffMapper(basis=basis, out_bands=cfg.out_bands).to(device)
    else:
        model = Raw2HSI(base_ch=cfg.base_ch, n_blocks=cfg.n_blocks, out_bands=cfg.out_bands).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def main(args):
    device = torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    ds = HyperObjectDataset(data_root=args.data_dir, track=1, train=False, submisison=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = load_model(args.ckpt, device, model_type=args.model, pca_basis_path=args.pca_basis)

    ids = []
    for batch in loader:
        x = batch["input"].float().to(device)
        sid = batch["id"][0]
        # TTA
        xs, invs = tta_transforms(x)
        preds = []
        with torch.no_grad():
            for x_aug, inv in zip(xs, invs):
                p = model(x_aug).clamp(0, 1)
                preds.append(inv(p))
        pred = torch.stack(preds, dim=0).mean(dim=0)
        pred_np = pred.squeeze(0).cpu().numpy()
        pred_hwc = np.transpose(pred_np, (1, 2, 0))
        np.savez_compressed(os.path.join(args.out_dir, f"{sid}.npz"), cube=pred_hwc)
        ids.append(sid)

    csv_path = os.path.join(args.out_dir, "submission.csv")
    pd.DataFrame({"id": ids, "prediction": 0}).to_csv(csv_path, index=False)
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.csv")
        for sid in ids:
            zf.write(os.path.join(args.out_dir, f"{sid}.npz"), arcname=f"{sid}.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/track1")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="submission_files_track1")
    p.add_argument("--zip_path", type=str, default="submission_track1.zip")
    p.add_argument("--model", type=str, default="linear", choices=["linear", "raw2hsi", "pca"])
    p.add_argument("--pca_basis", type=str, default=None)
    args = p.parse_args()
    main(args)

