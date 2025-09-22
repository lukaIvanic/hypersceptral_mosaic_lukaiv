import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.hyper_object import HyperObjectDataset
from datasets.base import JointTransform
from datasets.transform import random_flip

from baselines.pca_mapper import PCABasis, PCACoeffMapper
from trainer.trainer import Trainer
from trainer.losses import ReconLoss
from config.track1_cfg import TrainerCfg


def load_pca_basis(path: str, device: torch.device) -> PCABasis:
    data = np.load(path)
    return PCABasis(mean=data["mean"], components=data["components"]).to_torch(device)


def main(args):
    device = torch.device("cpu")

    ds_train = HyperObjectDataset(
        data_root=args.data_dir,
        track=1,
        train=True,
        transforms=JointTransform(random_flip),
    )
    ds_val = HyperObjectDataset(
        data_root=args.data_dir,
        track=1,
        train=False,
    )
    train_loader = DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(ds_val,   batch_size=2, shuffle=False, num_workers=0, pin_memory=False)

    cfg = TrainerCfg()
    basis = load_pca_basis(args.pca_basis, device)
    model = PCACoeffMapper(basis=basis, out_bands=cfg.out_bands)
    loss_fn = ReconLoss(lambda_sam=cfg.lambda_sam)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        cfg=cfg,
        device=device,
    )
    trainer.fit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/track1")
    p.add_argument("--pca_basis", type=str, required=True)
    args = p.parse_args()
    main(args)

