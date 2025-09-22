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
from config.track1_cfg import TrainerCfg

def _now_sync_cpu() -> float:
    # CPU timer
    return time.perf_counter()

def load_model(ckpt_path, device):
    print(f"[Load] Loading model from: {ckpt_path}")
    cfg = TrainerCfg()
    model = Raw2HSI(base_ch=cfg.base_ch, n_blocks=cfg.n_blocks, out_bands=cfg.out_bands).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        print("[Load] Detected Trainer checkpoint format. Using state['model'].")
        state = state["model"]
    else:
        print("[Load] Detected raw state_dict format.")
    model.load_state_dict(state)
    model.eval()
    print("[Load] Model ready (eval mode).\n")
    return model

def main(args):
    device = torch.device("cpu")  # run on CPU
    print(f"[Init] Using device: {device}")
    print(f"[Init] Output dir: {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Data] Building test dataset from: {args.data_dir}")
    ds_test = HyperObjectDataset(
        data_root=args.data_dir,  # expects .../track1 with test-original/{mosaic}
        track=1,
        train=False,
        submisison=True,  # use private test set (test_original) → 4 rows
    )
    print(f"[Data] Test samples: {len(ds_test)}")
    loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

    model = load_model(args.ckpt, device)

    ids = []
    steps_total = len(loader)
    print(f"[Run] Generating predictions for {steps_total} samples...\n")

    t_epoch0 = _now_sync_cpu()
    avg_step_ms = None

    for step_idx, batch in enumerate(loader, start=1):
        t_iter = _now_sync_cpu()

        # Unpack
        t = _now_sync_cpu()
        x = batch["input"].float().to(device)     # (1,1,H,W)
        sid = batch["id"][0]
        to_cpu_ms = (_now_sync_cpu() - t) * 1000.0

        # Inference
        t = _now_sync_cpu()
        with torch.no_grad():
            pred = model(x).clamp(0, 1)          # (1,61,H,W)
        fwd_ms = (_now_sync_cpu() - t) * 1000.0

        # Post-process + save
        t = _now_sync_cpu()
        pred_np = pred.squeeze(0).cpu().numpy()   # (61,H,W)
        pred_hwc = np.transpose(pred_np, (1, 2, 0))  # (H,W,61)
        out_path = os.path.join(args.out_dir, f"{sid}.npz")
        np.savez_compressed(out_path, cube=pred_hwc)
        save_ms = (_now_sync_cpu() - t) * 1000.0
        ids.append(sid)

        iter_ms = (_now_sync_cpu() - t_iter) * 1000.0
        avg_step_ms = iter_ms if avg_step_ms is None else (0.9 * avg_step_ms + 0.1 * iter_ms)
        remaining = steps_total - step_idx
        eta_s = (avg_step_ms / 1000.0) * remaining if avg_step_ms is not None else 0.0

        print(
            f"[Step {step_idx:03d}/{steps_total}] id={sid} | to_cpu {to_cpu_ms:.1f}ms  "
            f"fwd {fwd_ms:.1f}ms  save {save_ms:.1f}ms  iter {iter_ms:.1f}ms | "
            f"ETA ~ {eta_s/60.0:.2f} min"
        )

    dt = _now_sync_cpu() - t_epoch0
    print(f"\n[Run] Finished predictions in {dt/60.0:.2f} min. Writing submission.csv and zip...")

    # submission.csv (convention mirrors track2)
    csv_path = os.path.join(args.out_dir, "submission.csv")
    pd.DataFrame({"id": ids, "prediction": 0}).to_csv(csv_path, index=False)
    print(f"[Out] Wrote CSV: {csv_path} ({len(ids)} rows)")

    # zip it
    t = _now_sync_cpu()
    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="submission.csv")
        for sid in ids:
            npz_path = os.path.join(args.out_dir, f"{sid}.npz")
            if os.path.exists(npz_path):
                zf.write(npz_path, arcname=f"{sid}.npz")
    zip_ms = (_now_sync_cpu() - t) * 1000.0
    print(f"[Out] Submission zip: {args.zip_path} (zip time {zip_ms:.0f} ms)")
    print("[Done] You can now submit the zip to Kaggle.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/track1")
    p.add_argument("--ckpt", type=str, required=True, help="Path to model_best.pt or model_last.pt")
    p.add_argument("--out_dir", type=str, default="submission_files_track1")
    p.add_argument("--zip_path", type=str, default="submission_track1.zip")
    args = p.parse_args()
    main(args)