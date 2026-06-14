from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict
import logging
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import create_dataloaders
from .metrics import aggregate
from .models.simple_cnn.model import SimpleCNN

try:
    import psutil  # type: ignore[import-error]
except ImportError:  # pragma: no cover
    psutil = None

logger = logging.getLogger(__name__)
_PROCESS = psutil.Process() if psutil is not None else None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    log_interval: int,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    num_samples = 0
    steps_total = len(loader)
    timings = {
        "io": 0.0,
        "preprocess": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "interp": 0.0,
        "ram_mb": 0.0,
    }
    prev_time = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        io_time = time.perf_counter() - prev_time
        preprocess_start = time.perf_counter()
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        preprocess_time = time.perf_counter() - preprocess_start
        batch_size = inputs.size(0)

        optimizer.zero_grad(set_to_none=True)
        forward_start = time.perf_counter()
        preds = model(inputs)
        forward_time = time.perf_counter() - forward_start
        interp_time = getattr(model, "last_interp_time", 0.0)

        backward_start = time.perf_counter()
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        backward_time = time.perf_counter() - backward_start

        running_loss += loss.item() * batch_size
        num_samples += batch_size

        timings["io"] += io_time
        timings["preprocess"] += preprocess_time
        timings["forward"] += forward_time
        timings["backward"] += backward_time
        timings["interp"] += interp_time
        if _PROCESS is not None:
            ram_mb = _PROCESS.memory_info().rss / (1024 ** 2)
            timings["ram_mb"] += ram_mb
        else:
            ram_mb = None

        logger.debug(
            "epoch %d step %d/%d | io=%.2f ms | preprocess=%.2f ms | forward=%.2f ms | "
            "backward=%.2f ms | interp=%.2f ms%s",
            epoch,
            step,
            steps_total,
            io_time * 1e3,
            preprocess_time * 1e3,
            forward_time * 1e3,
            backward_time * 1e3,
            interp_time * 1e3,
            "" if ram_mb is None else f" | ram={ram_mb:.1f} MB",
        )

        if step % log_interval == 0:
            avg_loss = running_loss / max(num_samples, 1)
            logger.info(
                "[Train] Epoch %d Step %d/%d | loss=%.4f",
                epoch,
                step,
                steps_total,
                avg_loss,
            )

        prev_time = time.perf_counter()

    avg_times = {k: (v / max(steps_total, 1)) for k, v in timings.items()}
    avg_times["loss"] = running_loss / max(num_samples, 1)
    return avg_times


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    counts = 0
    metrics_sum = {k: 0.0 for k in ("mae", "mse", "psnr", "sam")}

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        batch_size = inputs.size(0)

        preds = model(inputs)
        loss = loss_fn(preds, targets)

        total_loss += loss.item() * batch_size
        counts += batch_size

        batch_metrics = aggregate(preds, targets, metrics=("mae", "mse", "psnr", "sam"))
        for key, value in batch_metrics.items():
            metrics_sum[key] += value * batch_size

    results = {k: v / max(counts, 1) for k, v in metrics_sum.items()}
    results["loss"] = total_loss / max(counts, 1)
    return results


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: Path) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)
    logger.info("[Checkpoint] Saved to %s", path)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = TrainConfig()

    if args.data_root is not None:
        cfg.data_root = Path(args.data_root)
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.prefetch_factor is not None:
        cfg.prefetch_factor = args.prefetch_factor
    if args.cache_dir is not None:
        cfg.cache_dir = None if args.cache_dir.lower() in {"none", ""} else Path(args.cache_dir)
    if args.no_cache:
        cfg.cache_dir = None
    if args.no_write_cache:
        cfg.write_cache = False
    if args.device is not None:
        cfg.device = args.device
    if args.hidden_channels is not None:
        cfg.hidden_channels = args.hidden_channels
    if args.log_interval is not None:
        cfg.log_interval = args.log_interval
    if args.resize_to is not None:
        cfg.resize_to = args.resize_to if args.resize_to > 0 else None

    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    model_runs_root = Path(__file__).resolve().parent / "models" / "simple_cnn" / "runs"
    run_dir = model_runs_root / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resize_info = cfg.resize_to if cfg.resize_to is not None else "native"
    cache_info = cfg.cache_dir if cfg.cache_dir is not None else "disabled"
    cache_mode = "rw" if cfg.write_cache else "ro"
    logger.info(
        "[Setup] device=%s | data_root=%s | resize=%s | cache=%s (%s) | run_dir=%s",
        device,
        cfg.data_root,
        resize_info,
        cache_info,
        cache_mode,
        run_dir,
    )
    if _PROCESS is None:
        logger.warning("psutil not available; RAM telemetry disabled.")

    train_loader, val_loader = create_dataloaders(cfg)

    model = SimpleCNN(
        in_channels=cfg.input_channels,
        out_channels=cfg.output_channels,
        coarse_channels=cfg.coarse_output_channels,
        hidden_channels=cfg.hidden_channels,
        train_resolution=cfg.resize_to,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.L1Loss()

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, cfg.log_interval, epoch
        )
        logger.info("[Train] Epoch %d done | loss=%.4f", epoch, train_stats["loss"])
        msg = (
            "Epoch %d timing (ms): io=%.2f | preprocess=%.2f | forward=%.2f | "
            "backward=%.2f | interp=%.2f"
        )
        args = [
            epoch,
            train_stats["io"] * 1e3,
            train_stats["preprocess"] * 1e3,
            train_stats["forward"] * 1e3,
            train_stats["backward"] * 1e3,
            train_stats["interp"] * 1e3,
        ]
        if _PROCESS is not None and train_stats.get("ram_mb") is not None:
            msg += " | ram=%.1f MB"
            args.append(train_stats["ram_mb"])
        logger.info(msg, *args)

        if epoch % cfg.val_interval == 0:
            val_metrics = evaluate(model, val_loader, device, loss_fn)
            metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            logger.info("[Val] Epoch %d | %s", epoch, metrics_str)

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(model, optimizer, epoch, ckpt_dir / "model_best.pt")

        if epoch % cfg.checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch, ckpt_dir / f"model_epoch_{epoch:03d}.pt")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Simple Track 1 model")
    parser.add_argument("--data-root", type=str, default=None, help="Path to data/track1 directory")
    parser.add_argument("--run-name", type=str, default=None, help="Name of the run subdirectory under the model's runs folder")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument(
        "--resize-to",
        type=int,
        default=None,
        help="Optional spatial size to resize inputs/targets to (e.g., 64).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="DataLoader prefetch factor (requires num_workers > 0).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Base directory for resized cache (use 'none' to disable).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading from disk cache even if available.",
    )
    parser.add_argument(
        "--no-write-cache",
        action="store_true",
        help="Do not write resized samples to disk cache.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level (e.g., DEBUG, INFO, WARNING).",
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
