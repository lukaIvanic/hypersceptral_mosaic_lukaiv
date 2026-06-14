from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import Track1Dataset
from .metrics import aggregate, list_available_metrics
from .models.simple_cnn.model import SimpleCNN


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    metric_names: list[str],
    progress_updates: int = 5,
) -> dict:
    model.eval()
    total_loss = 0.0
    count = 0
    metric_accum = {name: 0.0 for name in metric_names}
    total_batches = len(loader)
    progress_every: int | None
    if progress_updates <= 0 or total_batches == 0:
        progress_every = None
    else:
        progress_every = max(1, total_batches // progress_updates)

    for batch_idx, batch in enumerate(loader, start=1):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        batch_size = inputs.size(0)

        preds = model(inputs)
        loss = loss_fn(preds, targets)

        total_loss += loss.item() * batch_size
        count += batch_size

        metrics = aggregate(preds, targets, metric_names)
        for key, value in metrics.items():
            metric_accum[key] += value * batch_size

        if progress_every is not None and (
            batch_idx % progress_every == 0 or batch_idx == total_batches
        ):
            running_metrics = {k: metric_accum[k] / max(count, 1) for k in metric_names}
            running_metrics["loss"] = total_loss / max(count, 1)
            summary = " | ".join(f"{key}={value:.4f}" for key, value in running_metrics.items())
            print(f"[Eval] {batch_idx}/{total_batches} | {summary}")

    results = {k: v / max(count, 1) for k, v in metric_accum.items()}
    results["loss"] = total_loss / max(count, 1)
    return results


def load_model(ckpt_path: Path, cfg: TrainConfig, device: torch.device) -> nn.Module:
    model = SimpleCNN(
        in_channels=cfg.input_channels,
        out_channels=cfg.output_channels,
        coarse_channels=cfg.coarse_output_channels,
        hidden_channels=cfg.hidden_channels,
        train_resolution=cfg.resize_to,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def main(args: argparse.Namespace) -> None:
    cfg = TrainConfig()
    cfg.resize_to = None  # default to native resolution for evaluation
    if args.data_root is not None:
        cfg.data_root = Path(args.data_root)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.prefetch_factor is not None:
        cfg.prefetch_factor = args.prefetch_factor
    if args.device is not None:
        cfg.device = args.device
    if args.hidden_channels is not None:
        cfg.hidden_channels = args.hidden_channels
    if args.resize_to is not None:
        cfg.resize_to = args.resize_to if args.resize_to > 0 else None
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.cache_dir is not None:
        cfg.cache_dir = None if args.cache_dir.lower() in {"none", ""} else Path(args.cache_dir)
    if args.no_cache:
        cfg.cache_dir = None
    if args.no_write_cache:
        cfg.write_cache = False

    device = torch.device(cfg.device)

    dataset = Track1Dataset(
        root=cfg.data_root,
        split="val",
        resize_to=cfg.resize_to,
        cache_dir=cfg.cache_dir,
        write_cache=cfg.write_cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        model_runs_root = Path(__file__).resolve().parent / "models" / "simple_cnn" / "runs"
        run_dir = model_runs_root / cfg.run_name / "checkpoints"
        ckpt_path = run_dir / "model_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Pass --checkpoint explicitly.")

    model = load_model(ckpt_path, cfg, device)
    loss_fn = nn.L1Loss()
    metrics_requested = args.metrics.split(",") if args.metrics else list_available_metrics()
    metrics_requested = [m.strip().lower() for m in metrics_requested if m.strip()]
    if not metrics_requested:
        metrics_requested = list_available_metrics()
    available = set(list_available_metrics())
    unknown = [m for m in metrics_requested if m not in available]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available: {sorted(available)}")

    resize_info = cfg.resize_to if cfg.resize_to is not None else "native"
    cache_info = cfg.cache_dir if cfg.cache_dir is not None else "disabled"
    cache_mode = "rw" if cfg.write_cache else "ro"
    print(f"[Eval] Evaluating {ckpt_path} on {len(dataset)} samples | resize={resize_info} | cache={cache_info} ({cache_mode})")
    metrics = run_evaluation(
        model,
        loader,
        device,
        loss_fn,
        metrics_requested,
        progress_updates=args.progress_updates,
    )
    for key in metrics_requested + ["loss"]:
        value = metrics[key]
        print(f"{key:>8s}: {value:.4f}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Track 1 checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt). If omitted, uses --run-name/model_best.pt")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="DataLoader prefetch factor (requires num_workers > 0).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument(
        "--resize-to",
        type=int,
        default=0,
        help="Optional spatial size to resize inputs/targets to (e.g., 64). Use 0 to keep native size.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the run under src/models/simple_cnn/runs/ (used when --checkpoint is omitted).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="sam,sid,ergas,psnr_srgb,ssim_srgb,deltae00",
        help="Comma-separated list of metrics to compute. Available: "
             + ",".join(list_available_metrics()),
    )
    parser.add_argument(
        "--progress-updates",
        type=int,
        default=5,
        help="How many evenly spaced progress updates to print during evaluation (set <=0 to disable).",
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
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
