from torch.utils.data import DataLoader
from baselines.raw2hsi import Raw2HSI
from baselines.linear_mapper import LinearRaw2HSI
from trainer.losses import ReconLoss, ReconLossWithColorSmooth
from trainer.trainer import Trainer

from datasets.hyper_object import HyperObjectDataset
from datasets.pairing import ModalitySpec
from datasets.base import JointTransform
from datasets.transform import random_flip
import os

from config.track1_cfg import TrainerCfg

import torch 


ds_train = HyperObjectDataset(
    data_root="data/track1",
    track=1,  # 1 for mosaic, 2 for rgb_2
    train=True,
    transforms=JointTransform(random_flip),
)

ds_val = HyperObjectDataset(
    data_root="data/track1",
    track=1,  # 1 for mosaic, 2 for rgb_2
    train=False,
)

# ds_train / ds_val should yield dict with keys: "mosaic": (N,1,H,W), "cube": (N,61,H,W)
torch.set_num_threads(max(os.cpu_count() or 1, 1))
train_loader = DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(ds_val,   batch_size=2, shuffle=False, num_workers=0, pin_memory=False)

cfg = TrainerCfg()

# Choose a simple CPU-friendly default: LinearRaw2HSI
# You can switch back to Raw2HSI by uncommenting below
model = LinearRaw2HSI(out_bands=cfg.out_bands)
# model = Raw2HSI(base_ch=cfg.base_ch, n_blocks=cfg.n_blocks, out_bands=cfg.out_bands)
loss_fn = ReconLoss(lambda_sam=cfg.lambda_sam)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    cfg=cfg,
)

trainer.fit()
