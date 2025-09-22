import argparse
import json
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader, Subset

from datasets.hyper_object import HyperObjectDataset
from datasets.base import JointTransform
from datasets.transform import random_flip
from trainer.trainer import Trainer
from trainer.losses import ReconLoss
from baselines.linear_mapper import LinearRaw2HSI
from config.track1_cfg import TrainerCfg


def main(args):
    with open(args.splits_json, "r") as f:
        folds = json.load(f)
    all_ids = set()
    for k in folds:
        all_ids |= set(folds[k])

    base_ds = HyperObjectDataset(data_root=args.data_dir, track=1, train=True, transforms=JointTransform(random_flip))
    id_to_idx = {sid: i for i, sid in enumerate(base_ds.ids)}

    for k, val_ids in folds.items():
        val_idx = [id_to_idx[sid] for sid in val_ids if sid in id_to_idx]
        train_idx = [i for i in range(len(base_ds)) if base_ds.ids[i] not in val_ids]

        ds_train = Subset(base_ds, train_idx)
        ds_val_full = HyperObjectDataset(data_root=args.data_dir, track=1, train=False)
        loader_train = DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=0)
        loader_val   = DataLoader(ds_val_full, batch_size=2, shuffle=False, num_workers=0)

        cfg = TrainerCfg(out_dir=str(Path(args.out_dir) / k))
        model = LinearRaw2HSI(out_bands=cfg.out_bands)
        loss_fn = ReconLoss(lambda_sam=cfg.lambda_sam)

        trainer = Trainer(model=model, train_loader=loader_train, val_loader=loader_val, loss_fn=loss_fn, cfg=cfg, device=torch.device("cpu"))
        trainer.fit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/track1")
    p.add_argument("--splits_json", type=str, default="runs/track1/splits/kfold_3.json")
    p.add_argument("--out_dir", type=str, default="runs/track1/kfold")
    args = p.parse_args()
    main(args)

