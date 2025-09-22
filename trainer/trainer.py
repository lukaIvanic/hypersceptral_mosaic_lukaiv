from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import csv
import time
import numpy as np
import colour

import torch
from torch.utils.data import DataLoader

# ---- bring your stuff ----
# - loss: ReconLoss (L1 + SAM) or any callable loss(pred, target, mask=None)
# - metrics: rmse/sam/sid/ergas/psnr/ssim
# - render: function to convert HSI cube -> sRGB under D65
from .losses import ReconLoss

from config.track1_cfg import TrainerCfg
from utils.helpers import _to_hwc
from utils.metrics import sam, sid, ergas
from utils.metrics import psnr, ssim
from utils.visualizations import render_srgb_preview  # returns HxWx3 float [0,1]

class Trainer:
    """
    Generic trainer for RAW mosaic -> HSI models.

    Expects each batch dict with:
      - "mosaic": (N,1,H,W) float in [0,1]
      - "cube":   (N,C,H,W) float in [0,1] (C=61 for your case)
      - Optional "mask": (N,1,H,W) bool/float (ROI), used only for metrics if present
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cfg: TrainerCfg = TrainerCfg(),
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader   
        self.device = device 

        self.model = model
        self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler() if (self.cfg.amp and self.device.type == "cuda") else None
        self.loss_fn = loss_fn if loss_fn is not None else ReconLoss(lambda_sam=0.1)
        
        self.wl_nm = self.cfg.wl_61  # wavelength vector for rendering
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        if self.cfg.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.eta_min)
        else:
            self.scheduler = None

        # I/O
        self.out_dir = Path(self.cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_best = self.out_dir / "model_best.pt"
        self.ckpt_last = self.out_dir / "model_last.pt"
        self.log_csv = self.out_dir / self.cfg.log_csv_name

        # CSV header
        if not self.log_csv.exists():
            with open(self.log_csv, "w", newline="") as f:
                w = csv.writer(f)
                header = [
                    "epoch", "lr", "train_loss", "val_loss",
                    "SAM_deg", "SID", "ERGAS",
                    "PSNR_dB", "SSIM", 
                ]
                w.writerow(header)

        # Timing CSVs
        self.train_timing_csv = self.out_dir / "train_timing.csv"
        if not self.train_timing_csv.exists():
            with open(self.train_timing_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch", "step", "steps_total", "batch_size",
                    "to_device_ms", "forward_ms", "loss_ms", "backward_ms", "optim_ms",
                    "iter_ms", "samples_per_s",
                ])
        self.val_timing_csv = self.out_dir / "val_timing.csv"
        if not self.val_timing_csv.exists():
            with open(self.val_timing_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch", "step", "steps_total", "batch_size",
                    "to_device_ms", "forward_ms", "metrics_ms", "iter_ms",
                ])

        self.best_val = float("inf")


    def _sync_cuda(self) -> None:
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _now(self) -> float:
        # High-res timer with CUDA sync for accurate GPU timings
        self._sync_cuda()
        return time.perf_counter()

    def _append_csv(self, path: Path, row: List[Any]) -> None:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _forward_loss(self, input_img: torch.Tensor, output_cube: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scaler is None:
            pred = self.model(input_img)
            loss = self.loss_fn(pred, output_cube)
            return pred, loss
        else:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                pred = self.model(input_img)
                loss = self.loss_fn(pred, output_cube)
            return pred, loss

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        n_samples = 0
        t0 = time.time()

        steps_total = len(self.train_loader)
        print(f"[Train] Epoch {epoch} | steps: {steps_total}")
        sum_to_dev = sum_fwd = sum_loss = sum_bwd = sum_opt = sum_iter = 0.0

        for step_idx, batch in enumerate(self.train_loader, start=1):
            iter_start = self._now()

            # To device timing
            t = self._now()
            input_img  :   torch.Tensor = batch["input"].to(self.device, non_blocking=True)     # (N,c(1 or 3),H,W)
            output_cube:   torch.Tensor = batch["output"].to(self.device, non_blocking=True)    # (N,C,H,W)
            to_device_ms = (self._now() - t) * 1000.0

            bs = int(input_img.size(0))

            # Zero grad
            self.optimizer.zero_grad(set_to_none=True)

            # Forward + loss timing
            t = self._now()
            pred, loss = self._forward_loss(input_img, output_cube)
            forward_ms = (self._now() - t) * 1000.0

            # Explicit loss read (negligible, but measured)
            t = self._now()
            loss_value = float(loss.item())
            loss_ms = (self._now() - t) * 1000.0

            # Backward + optimizer timing
            if self.scaler is None:
                t = self._now(); loss.backward(); backward_ms = (self._now() - t) * 1000.0
                t = self._now(); self.optimizer.step(); optim_ms = (self._now() - t) * 1000.0
            else:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    pass  # already forward in autocast within _forward_loss
                t = self._now(); self.scaler.scale(loss).backward(); backward_ms = (self._now() - t) * 1000.0
                t = self._now(); self.scaler.step(self.optimizer); self.scaler.update(); optim_ms = (self._now() - t) * 1000.0

            iter_ms = (self._now() - iter_start) * 1000.0
            samples_per_s = (bs / (iter_ms / 1000.0)) if iter_ms > 0 else float("inf")

            running += loss_value * bs
            n_samples += bs

            # Accumulate for epoch averages
            sum_to_dev += to_device_ms; sum_fwd += forward_ms; sum_loss += loss_ms
            sum_bwd += backward_ms; sum_opt += optim_ms; sum_iter += iter_ms

            # CSV timing log per step
            self._append_csv(self.train_timing_csv, [
                epoch, step_idx, steps_total, bs,
                f"{to_device_ms:.3f}", f"{forward_ms:.3f}", f"{loss_ms:.3f}", f"{backward_ms:.3f}", f"{optim_ms:.3f}",
                f"{iter_ms:.3f}", f"{samples_per_s:.2f}",
            ])

            # Console print each step
            print(
                f"[Train] ep {epoch} step {step_idx}/{steps_total} | "
                f"to_dev {to_device_ms:.2f}ms  fwd {forward_ms:.2f}ms  "
                f"bwd {backward_ms:.2f}ms  opt {optim_ms:.2f}ms  "
                f"iter {iter_ms:.2f}ms  ips {samples_per_s:.1f}"
            )

        if self.scheduler is not None:
            t_sched = self._now(); self.scheduler.step(); sched_ms = (self._now() - t_sched) * 1000.0
            print(f"[Train] Epoch {epoch} scheduler.step() took {sched_ms:.2f} ms")

        avg = running / max(n_samples, 1)
        dt = time.time() - t0
        # Epoch-level averages
        if steps_total > 0:
            print(
                f"[Train] Epoch {epoch} done in {dt:.2f}s | "
                f"avg to_dev {sum_to_dev/steps_total:.2f}ms, fwd {sum_fwd/steps_total:.2f}ms, "
                f"bwd {sum_bwd/steps_total:.2f}ms, opt {sum_opt/steps_total:.2f}ms, iter {sum_iter/steps_total:.2f}ms"
            )
        return avg

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Returns dict with: val_loss, SAM_deg, SID, ERGAS, PSNR_dB, SSIM, (DeltaE00 optional).
        """
        self.model.eval()

        loss_sum = 0.0
        n_samples = 0


        sam_list: List[float] = []
        sid_list: List[float] = []
        erg_list: List[float] = []
        psnr_list: List[float] = []
        ssim_list: List[float] = []
        
        steps_total = len(self.val_loader)
        print(f"[Valid] Epoch {epoch} | steps: {steps_total}")


        
        for step_idx, batch in enumerate(self.val_loader, start=1):
            iter_start = self._now()
            t = self._now()
            input_img  :   torch.Tensor = batch["input"].to(self.device, non_blocking=True)
            output_cube:   torch.Tensor = batch["output"].to(self.device, non_blocking=True)
            to_device_ms = (self._now() - t) * 1000.0

            # forward (no grad)
            t = self._now()
            pred_cube = self.model(input_img).clamp(0, 1)
            loss = self.loss_fn(pred_cube, output_cube)
            forward_ms = (self._now() - t) * 1000.0

            loss_sum += float(loss.item()) * input_img.size(0)
            n_samples += input_img.size(0)

            # per-sample metrics
            t = self._now()

            for i in range(pred_cube.size(0)):
                # Optional mask support
                mask_np = None
                if "mask" in batch and batch["mask"] is not None:
                    m = batch["mask"][i].detach().cpu().numpy()
                    mask_np = m.squeeze() if m.ndim == 3 else m

                ref_np = output_cube[i].detach().cpu().numpy()
                est_np = pred_cube[i].detach().cpu().numpy()

                sam_val = sam(ref_np, est_np, reduction="mean", mask=mask_np)

                sid_val = sid(ref_np, est_np, reduction="mean", mask=mask_np)

                erg_val = ergas(ref_np, est_np, scale=1.0)

                psnr_val = psnr(ref_np, est_np, data_range=1.0, mask=mask_np)

                # ssim_val = ssim(ref_np, est_np, data_range=1.0, mask=mask_np)
                ssim_val = 0.0

                sam_list.append(sam_val)
                sid_list.append(sid_val)
                erg_list.append(erg_val)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

            metrics_ms = (self._now() - t) * 1000.0

            iter_ms = (self._now() - iter_start) * 1000.0

            # CSV timing log per step
            self._append_csv(self.val_timing_csv, [
                epoch, step_idx, steps_total, int(input_img.size(0)),
                f"{to_device_ms:.3f}", f"{forward_ms:.3f}", f"{metrics_ms:.3f}", f"{iter_ms:.3f}",
            ])

            # Console print each step
            print(
                f"[Valid] ep {epoch} step {step_idx}/{steps_total} | "
                f"to_dev {to_device_ms:.2f}ms  fwd {forward_ms:.2f}ms  metrics {metrics_ms:.2f}ms  iter {iter_ms:.2f}ms"
            )

        out: Dict[str, float] = {}
        out["val_loss"] = loss_sum / max(n_samples, 1)
        out["SAM_deg"]  = float(np.mean(sam_list)) if sam_list else float("nan")
        out["SID"]      = float(np.mean(sid_list)) if sid_list else float("nan")
        out["ERGAS"]    = float(np.mean(erg_list)) if erg_list else float("nan")
        out["PSNR_dB"]  = float(np.mean(psnr_list)) if psnr_list else float("nan")
        out["SSIM"]     = float(np.mean(ssim_list)) if ssim_list else float("nan")
        return out

    def _current_lr(self) -> float:
        if self.optimizer.param_groups:
            return float(self.optimizer.param_groups[0].get("lr", 0.0))
        return 0.0

    def _log_csv(self, epoch: int, train_loss: float, val_stats: Dict[str, float]):
        row = [
            epoch,
            self._current_lr(),
            train_loss,
            val_stats.get("val_loss", float("nan")),
            val_stats.get("SAM_deg", float("nan")),
            val_stats.get("SID", float("nan")),
            val_stats.get("ERGAS", float("nan")),
            val_stats.get("PSNR_dB", float("nan")),
            val_stats.get("SSIM", float("nan")),
        ]
        with open(self.log_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _print_epoch(self, epoch: int, train_loss: float, val_stats: Dict[str, float]):
        parts = [
            # --- Epoch and Learning Info ---
            f"[{epoch:03d}]",
            f"lr: {self._current_lr():.2e}",
            f"train: {train_loss:7.4f}",
            f"val: {val_stats.get('val_loss', float('nan')):7.4f} | ",

            # --- Core Reconstruction Metrics ---
            f"SAM(deg): {val_stats.get('SAM_deg', float('nan')):6.2f}",
            f"SID: {val_stats.get('SID', float('nan')):7.4f}",
            f"ERGAS: {val_stats.get('ERGAS', float('nan')):6.3f}",
            f"PSNR(dB): {val_stats.get('PSNR_dB', float('nan')):6.2f}",
            f"SSIM: {val_stats.get('SSIM', float('nan')):5.3f} | ",
        ]
        print("  ".join(parts))

    def _save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(state, self.ckpt_last)
        if is_best and self.cfg.save_best:
            torch.save(state, self.ckpt_best)
            print(f"[Saved BEST model @ epoch {epoch}] → {self.ckpt_best}")

    def fit(self):
        print(f"Start training for {self.cfg.epochs} epochs. Logs → {self.log_csv}")
        for ep in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch(ep)
            val_stats = self.validate(ep)
            self._print_epoch(ep, train_loss, val_stats)
            self._log_csv(ep, train_loss, val_stats)

            is_best = val_stats["val_loss"] < self.best_val
            if is_best:
                self.best_val = val_stats["val_loss"]
            self._save_checkpoint(ep, is_best)
