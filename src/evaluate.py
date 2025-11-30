from __future__ import annotations

import argparse
import copy
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import Track1Dataset
from .losses import CompositeSpectralLoss, LossWeights
from .metrics import aggregate, list_available_metrics
from .models.builder import create_model
from .utils.inference import run_model_with_resize
from .utils.tta import Transform, apply_tta, resolve_tta_mode, identity
from .utils.metrics.schema import (
    ACCURACY_METRIC_UNITS,
    SOURCE_EVALUATE,
    resolution_label,
    resolution_tag,
    units_from_defaults,
)
from .utils.metrics.storage import (
    ensure_epoch_record,
    load_metrics_history,
    save_metrics_history,
    utc_timestamp,
)

import shutil


def fix_best_checkpoint(run_dir: Path, loss_key: str = "loss", dry_run: bool = False) -> bool:
    """
    Check if model_best.pt corresponds to the epoch with lowest loss in metrics.json.
    If not, copy the correct epoch checkpoint to model_best.pt.
    
    This is useful when loss weights change mid-training, making earlier losses
    incomparable to later ones. After manually adjusting metrics.json, run this
    to sync model_best.pt with the actual best epoch.
    
    Args:
        run_dir: Path to the run directory containing checkpoints/ and metrics/
        loss_key: Which metric key to use for finding the best epoch (default: "loss")
        dry_run: If True, only report what would be done without making changes
    
    Returns:
        True if model_best.pt was updated (or would be updated in dry_run mode)
    """
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics" / "metrics.json"
    best_ckpt = ckpt_dir / "model_best.pt"
    
    if not metrics_path.exists():
        print(f"[FixBest] No metrics.json found at {metrics_path}")
        return False
    
    if not best_ckpt.exists():
        print(f"[FixBest] No model_best.pt found at {best_ckpt}")
        return False
    
    # Load metrics history
    history = load_metrics_history(metrics_path)
    if not history:
        print("[FixBest] metrics.json is empty")
        return False
    
    # Find epoch with minimum loss
    best_epoch = None
    best_loss = float("inf")
    for record in history:
        epoch = record.get("epoch")
        # Look for loss in val section first, then val_native
        val_data = record.get("val") or record.get("val_native") or {}
        loss_val = val_data.get(loss_key)
        if loss_val is not None and loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
    
    if best_epoch is None:
        print(f"[FixBest] Could not find any epoch with '{loss_key}' in metrics.json")
        return False
    
    # Check current model_best.pt epoch
    try:
        raw_state = torch.load(best_ckpt, map_location="cpu")
        current_best_epoch = raw_state.get("epoch") if isinstance(raw_state, dict) else None
    except Exception as exc:
        print(f"[FixBest] Failed to load model_best.pt: {exc}")
        return False
    
    if current_best_epoch == best_epoch:
        print(f"[FixBest] model_best.pt already corresponds to best epoch {best_epoch} (loss={best_loss:.6f})")
        return False
    
    # Find the checkpoint for the best epoch
    source_ckpt = ckpt_dir / f"model_epoch_{best_epoch:03d}.pt"
    if not source_ckpt.exists():
        # Try without zero-padding
        source_ckpt = ckpt_dir / f"model_epoch_{best_epoch}.pt"
    if not source_ckpt.exists():
        print(f"[FixBest] Cannot find checkpoint for epoch {best_epoch}")
        return False
    
    # Report and optionally fix
    print(
        f"[FixBest] Best epoch is {best_epoch} (loss={best_loss:.6f}), "
        f"but model_best.pt is from epoch {current_best_epoch}"
    )
    
    if dry_run:
        print(f"[FixBest] DRY RUN: Would copy {source_ckpt.name} -> model_best.pt")
    else:
        # Backup old best
        backup_path = ckpt_dir / f"model_best_backup_epoch{current_best_epoch}.pt"
        if not backup_path.exists():
            shutil.copy2(best_ckpt, backup_path)
            print(f"[FixBest] Backed up old model_best.pt to {backup_path.name}")
        
        # Copy the correct checkpoint
        shutil.copy2(source_ckpt, best_ckpt)
        print(f"[FixBest] Copied {source_ckpt.name} -> model_best.pt")
    
    return True


@dataclass
class EvaluationAccumulator:
    label: str
    epoch: int | None
    checkpoint_path: Path
    model: nn.Module
    metric_names: List[str]
    is_best: bool = False
    metrics_sum: Dict[str, float] = field(init=False)
    total_loss: float = 0.0
    total_samples: int = 0
    batches: int = 0

    def __post_init__(self) -> None:
        self.metrics_sum = {name: 0.0 for name in self.metric_names}
        self.model.eval()

    def update(
        self,
        loss_value: float,
        metrics: Dict[str, float],
        batch_size: int,
    ) -> None:
        self.total_loss += loss_value * batch_size
        self.total_samples += batch_size
        self.batches += 1
        for key, value in metrics.items():
            if key not in self.metrics_sum:
                # accommodate metrics discovered mid-run
                self.metrics_sum[key] = 0.0
            self.metrics_sum[key] += value * batch_size

    def average_loss(self) -> float:
        return self.total_loss / max(self.total_samples, 1)

    def averages(self) -> Dict[str, float]:
        denom = max(self.total_samples, 1)
        return {key: self.metrics_sum[key] / denom for key in self.metrics_sum}

    def finalize(self) -> Dict[str, float]:
        results = self.averages()
        results["loss"] = self.average_loss()
        return results

    def preview(self, metric_priority: List[str]) -> str:
        if self.total_samples == 0:
            return "no-data"
        parts = [f"loss={self.average_loss():.4f}"]
        for name in metric_priority:
            if name in self.metrics_sum:
                parts.append(f"{name}={self.metrics_sum[name] / self.total_samples:.4f}")
        return ", ".join(parts)


@torch.no_grad()
def run_multi_evaluation(
    accumulators: List[EvaluationAccumulator],
    dataset: Track1Dataset,
    loader_kwargs: Dict[str, Any],
    device: torch.device,
    loss_fn: nn.Module,
    metric_names: List[str],
    progress_updates: int = 5,
    inference_resize: int | None = None,
    upsample_metrics: bool = False,
    tta_transforms: List[Transform] | None = None,
) -> None:
    if not accumulators:
        return
    
    # Default to no TTA (single identity transform)
    if tta_transforms is None:
        tta_transforms = [identity]

    loader = DataLoader(dataset, **loader_kwargs)
    total_batches = len(loader)
    if total_batches == 0:
        print("[Eval] Warning: validation loader is empty.")
        return

    progress_every: int | None
    if progress_updates <= 0:
        progress_every = None
    else:
        progress_every = max(1, total_batches // progress_updates)

    preview_metrics = metric_names[:2] if metric_names else []
    logged_shapes: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    device_type = device.type

    prev_batch_end = time.perf_counter()
    for batch_idx, batch in enumerate(loader, start=1):
        batch_start = prev_batch_end
        after_load = time.perf_counter()
        load_ms = (after_load - batch_start) * 1000

        batch_inputs = batch["input"]
        batch_targets = batch["target"]
        input_shape = tuple(batch_inputs.shape)
        target_shape = tuple(batch_targets.shape)
        shape_pair = (input_shape, target_shape)
        if shape_pair not in logged_shapes:
            print(
                f"[Eval] Batch {batch_idx} shapes | input={input_shape} | target={target_shape}"
            )
            logged_shapes.add(shape_pair)

        inputs = batch_inputs.to(device, non_blocking=True)
        targets = batch_targets.to(device, non_blocking=True)
        batch_size = inputs.size(0)
        target_shape = batch_targets.shape[-2:]

        per_model_logs: List[
            Tuple[str, float, float, float, Dict[str, float], Dict[str, float]]
        ] = []
        for accumulator in accumulators:
            if device_type == "cuda":
                torch.cuda.synchronize(device)
            infer_start = time.perf_counter()
            # Apply TTA: run model on augmented inputs, inverse-transform, average
            preds = apply_tta(
                accumulator.model,
                inputs,
                tta_transforms,
                inference_resize,
                target_shape if upsample_metrics else None,
            )
            if device_type == "cuda":
                torch.cuda.synchronize(device)
            infer_ms = (time.perf_counter() - infer_start) * 1000
            if device_type == "cuda":
                torch.cuda.synchronize(device)
            metrics_start = time.perf_counter()
            loss = loss_fn(preds, targets)
            metrics = aggregate(preds, targets, metric_names, include_timings=True)
            if device_type == "cuda":
                torch.cuda.synchronize(device)
            metrics_ms = (time.perf_counter() - metrics_start) * 1000
            timings = metrics.pop("_timings", {}) if isinstance(metrics, dict) else {}
            accumulator.update(loss.item(), metrics, batch_size)
            per_model_logs.append(
                (
                    accumulator.label,
                    infer_ms,
                    metrics_ms,
                    float(loss.item()),
                    metrics,
                    timings,
                )
            )

        batch_end = time.perf_counter()
        batch_total_ms = (batch_end - batch_start) * 1000

        if progress_every is not None and (
            batch_idx % progress_every == 0 or batch_idx == total_batches
        ):
            print(
                f"[Eval] Batch {batch_idx}/{total_batches} | load={load_ms:.1f} ms | "
                f"total={batch_total_ms:.1f} ms"
            )
            for label, infer_ms, metrics_ms, loss_value, metrics, timings in per_model_logs:
                metric_bits = [
                    f"{name}={metrics[name]:.4f}"
                    for name in preview_metrics
                    if name in metrics
                ]
                timing_bits = [f"{name}={timings[name]:.1f} ms" for name in sorted(timings)]
                timing_text = " | ".join(timing_bits)
                metric_text = " | ".join(metric_bits)
                extra = ""
                if metric_text:
                    extra += f" | {metric_text}"
                if timing_text:
                    extra += f" | timings: {timing_text}"
                print(
                    f"[Eval][{label}] {batch_idx}/{total_batches} | "
                    f"infer={infer_ms:.1f} ms | metrics={metrics_ms:.1f} ms | "
                    f"loss={loss_value:.4f}{extra}"
                )

        prev_batch_end = batch_end


def infer_epoch_from_path(path: Path) -> int | None:
    match = re.search(r"epoch_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return None


def chunked(items: List[Any], size: int):
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def _remap_mst_feedforward_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remap old MST++ FeedForward keys to new format for backward compatibility.
    
    Old format (nn.Sequential):  fn.net.0.weight, fn.net.2.weight, fn.net.4.weight
    New format (individual):     fn.conv1.weight, fn.conv2.weight, fn.conv3.weight
    """
    key_map = {
        ".fn.net.0.": ".fn.conv1.",
        ".fn.net.2.": ".fn.conv2.",
        ".fn.net.4.": ".fn.conv3.",
    }
    remapped = {}
    num_remapped = 0
    for key, value in state_dict.items():
        new_key = key
        for old_pattern, new_pattern in key_map.items():
            if old_pattern in key:
                new_key = key.replace(old_pattern, new_pattern)
                num_remapped += 1
                break
        remapped[new_key] = value
    if num_remapped > 0:
        print(f"[Eval] Remapped {num_remapped} legacy MST++ FeedForward keys for compatibility.")
    return remapped


def load_model(
    ckpt_path: Path,
    cfg: TrainConfig,
    device: torch.device,
    *,
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    raw_state = torch.load(ckpt_path, map_location=device)
    state_dict = raw_state
    variant_from_ckpt: str | None = None
    epoch_from_ckpt: int | None = None
    if isinstance(raw_state, dict):
        variant_from_ckpt = raw_state.get("variant")
        epoch_from_ckpt = raw_state.get("epoch")
        state_dict = raw_state.get("model", raw_state)
    
    # Remap legacy MST++ FeedForward keys if present
    state_dict = _remap_mst_feedforward_keys(state_dict)

    variant = cfg.model_variant
    if variant_from_ckpt and variant_from_ckpt.lower() != variant.lower():
        print(f"[Eval] Checkpoint variant '{variant_from_ckpt}' overrides requested '{variant}'.")
        variant = variant_from_ckpt
        cfg.model_variant = variant

    model = create_model(
        variant,
        in_channels=cfg.input_channels,
        out_channels=cfg.output_channels,
        coarse_channels=cfg.coarse_output_channels,
        hidden_channels=cfg.hidden_channels,
        train_resolution=cfg.resize_to,
        unet_base_channels=cfg.unet_base_channels,
        latent_channels=cfg.latent_channels,
        encoder_depth=cfg.encoder_depth,
        use_residual_head=cfg.use_residual_head,
        use_spectral_conv=cfg.use_spectral_conv,
        spectral_conv_kernel_size=cfg.spectral_conv_kernel_size,
        decoder_dropout=cfg.decoder_dropout,
        stochastic_depth_p=cfg.stochastic_depth_p,
        use_bottleneck_attention=cfg.use_bottleneck_attention,
        conv_kernel_size=cfg.conv_kernel_size,
        norm_type=cfg.norm_type,
        use_raw_input_skip=cfg.use_raw_input_skip,
    ).to(device)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    info: Dict[str, Any] = {
        "epoch": epoch_from_ckpt,
        "checkpoint_variant": variant_from_ckpt,
        "loaded_variant": getattr(model, "variant_name", cfg.model_variant),
        "checkpoint_path": str(ckpt_path),
    }
    return model, info


class EnsembleModel(nn.Module):
    """
    Wrapper that holds multiple models and averages their predictions.
    
    All ensemble members must have the same architecture and input/output shapes.
    Predictions are averaged element-wise across all members.
    """
    
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        if not models:
            raise ValueError("EnsembleModel requires at least one model.")
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all models and average predictions."""
        # Run first model to get output shape
        preds = self.models[0](x)
        
        # Accumulate predictions from remaining models
        for model in self.models[1:]:
            preds = preds + model(x)
        
        # Average
        return preds / self.num_models
    
    def eval(self):
        """Set all models to eval mode."""
        for model in self.models:
            model.eval()
        return super().eval()
    
    def train(self, mode: bool = True):
        """Set all models to train mode (usually not needed for ensemble inference)."""
        for model in self.models:
            model.train(mode)
        return super().train(mode)


def load_ensemble_models(
    checkpoint_paths: List[Path],
    cfg: TrainConfig,
    device: torch.device,
    *,
    strict: bool = True,
) -> Tuple[EnsembleModel, Dict[str, Any]]:
    """
    Load multiple checkpoints and wrap them in an EnsembleModel.
    
    All checkpoints must be the same architecture variant.
    
    Args:
        checkpoint_paths: List of paths to checkpoint files.
        cfg: Training configuration (used for model architecture).
        device: Device to load models on.
        strict: Whether to strictly enforce state_dict matching.
    
    Returns:
        Tuple of (EnsembleModel, info_dict) where info_dict contains
        metadata about the loaded ensemble.
    """
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required for ensemble.")
    
    models: List[nn.Module] = []
    epochs: List[int | None] = []
    loaded_variant: str | None = None
    
    for idx, ckpt_path in enumerate(checkpoint_paths):
        model_cfg = copy.deepcopy(cfg)
        model, info = load_model(ckpt_path, model_cfg, device, strict=strict)
        
        # Verify all models have the same variant
        this_variant = info.get("loaded_variant", cfg.model_variant)
        if loaded_variant is None:
            loaded_variant = this_variant
        elif this_variant.lower() != loaded_variant.lower():
            raise ValueError(
                f"Ensemble checkpoint {idx + 1} has variant '{this_variant}' "
                f"but expected '{loaded_variant}'. All ensemble members must be the same architecture."
            )
        
        models.append(model)
        epochs.append(info.get("epoch"))
        
        num_params = sum(p.numel() for p in model.parameters())
        print(
            f"[Ensemble] Loaded member {idx + 1}/{len(checkpoint_paths)}: "
            f"{ckpt_path.name} ({num_params / 1e6:.2f}M params)"
        )
    
    ensemble = EnsembleModel(models)
    ensemble.eval()
    
    # Compute total params (same architecture, so just multiply)
    total_params = sum(p.numel() for p in ensemble.parameters())
    print(
        f"[Ensemble] Created ensemble with {len(models)} members | "
        f"total params: {total_params / 1e6:.2f}M"
    )
    
    info: Dict[str, Any] = {
        "epoch": epochs,
        "num_members": len(models),
        "loaded_variant": loaded_variant,
        "checkpoint_paths": [str(p) for p in checkpoint_paths],
    }
    return ensemble, info


def main(args: argparse.Namespace) -> None:
    cfg = TrainConfig()
    default_resize = cfg.resize_to
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
    if getattr(args, "resolution", None):
        if args.resolution == "native":
            cfg.resize_to = None
        elif args.resolution == "resized" and cfg.resize_to is None:
            cfg.resize_to = default_resize
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.cache_dir is not None:
        cfg.cache_dir = None if args.cache_dir.lower() in {"none", ""} else Path(args.cache_dir)
    if args.no_cache:
        cfg.cache_dir = None
    if args.no_write_cache:
        cfg.write_cache = False
    if getattr(args, "lambda_l1", None) is not None:
        cfg.lambda_l1 = args.lambda_l1
    if getattr(args, "lambda_sam", None) is not None:
        cfg.lambda_sam = args.lambda_sam
    if getattr(args, "lambda_sid", None) is not None:
        cfg.lambda_sid = args.lambda_sid
    if getattr(args, "lambda_srgb_l1", None) is not None:
        cfg.lambda_srgb_l1 = args.lambda_srgb_l1
    if getattr(args, "lambda_srgb_ssim", None) is not None:
        cfg.lambda_srgb_ssim = args.lambda_srgb_ssim
    if getattr(args, "lambda_ergas", None) is not None:
        cfg.lambda_ergas = args.lambda_ergas
    if getattr(args, "model_variant", None) is not None:
        cfg.model_variant = args.model_variant.lower()
    if getattr(args, "unet_base_channels", None) is not None:
        cfg.unet_base_channels = args.unet_base_channels
    if getattr(args, "latent_channels", None) is not None:
        cfg.latent_channels = args.latent_channels
    if getattr(args, "encoder_depth", None) is not None:
        cfg.encoder_depth = max(1, args.encoder_depth)
    if getattr(args, "coarse_channels", None) is not None:
        cfg.coarse_output_channels = max(1, args.coarse_channels)
    if getattr(args, "conv_kernel_size", None) is not None:
        cfg.conv_kernel_size = max(1, int(args.conv_kernel_size))
    if getattr(args, "norm_type", None) is not None:
        cfg.norm_type = args.norm_type.lower()
    if getattr(args, "use_residual_head", False):
        cfg.use_residual_head = True
    if getattr(args, "use_spectral_conv", False):
        cfg.use_spectral_conv = True
    if getattr(args, "spectral_conv_kernel_size", None) is not None:
        cfg.spectral_conv_kernel_size = max(1, int(args.spectral_conv_kernel_size))
    if getattr(args, "decoder_dropout", None) is not None:
        cfg.decoder_dropout = max(0.0, float(args.decoder_dropout))
    if getattr(args, "stochastic_depth_p", None) is not None:
        cfg.stochastic_depth_p = max(0.0, float(args.stochastic_depth_p))
    if getattr(args, "use_bottleneck_attention", False):
        cfg.use_bottleneck_attention = True
    if getattr(args, "use_raw_input_skip", False):
        cfg.use_raw_input_skip = True

    device = torch.device(cfg.device)

    upsample_metrics = bool(getattr(args, "upsample_metrics", False))
    inference_resize = cfg.resize_to
    dataset_resize = None if upsample_metrics else inference_resize

    dataset = Track1Dataset(
        root=cfg.data_root,
        split="val",
        resize_to=dataset_resize,
        cache_dir=cfg.cache_dir,
        write_cache=cfg.write_cache,
    )
    loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if cfg.prefetch_factor is not None and cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor

    model_runs_root = Path(__file__).resolve().parent / "models" / "simple_cnn" / "runs"
    run_dir: Path | None = model_runs_root / cfg.run_name if args.run_name is not None else None

    # Handle --fix-best-checkpoint / --fix-best-dry-run before loading checkpoints
    fix_best = getattr(args, "fix_best_checkpoint", False)
    fix_best_dry = getattr(args, "fix_best_dry_run", False)
    if (fix_best or fix_best_dry) and run_dir is not None:
        fixed = fix_best_checkpoint(run_dir, dry_run=fix_best_dry)
        if fix_best_dry and fixed:
            print("[Eval] Dry run complete. Use --fix-best-checkpoint to apply changes.")
            return
        elif fix_best and not fixed:
            print("[Eval] No changes needed to model_best.pt.")

    # Check for ensemble mode
    ensemble_checkpoints_arg = getattr(args, "ensemble_checkpoints", None)
    use_ensemble = ensemble_checkpoints_arg is not None and ensemble_checkpoints_arg.strip()
    
    if use_ensemble:
        # Parse comma-separated checkpoint paths
        ensemble_paths = [
            Path(p.strip()).expanduser().resolve()
            for p in ensemble_checkpoints_arg.split(",")
            if p.strip()
        ]
        if len(ensemble_paths) < 2:
            raise ValueError("--ensemble-checkpoints requires at least 2 comma-separated checkpoint paths.")
        for p in ensemble_paths:
            if not p.exists():
                raise FileNotFoundError(f"Ensemble checkpoint not found: {p}")
        # Validate incompatible flags
        if args.checkpoint is not None:
            raise ValueError("--ensemble-checkpoints cannot be combined with --checkpoint.")
        if getattr(args, "all_checkpoints", False):
            raise ValueError("--ensemble-checkpoints cannot be combined with --all-checkpoints.")

    checkpoint_jobs: List[Tuple[Path, bool]] = []
    max_checkpoints = args.max_checkpoints if args.max_checkpoints and args.max_checkpoints > 0 else None

    if getattr(args, "all_checkpoints", False):
        if args.checkpoint is not None:
            raise ValueError("--all-checkpoints cannot be combined with --checkpoint.")
        if run_dir is None:
            raise ValueError("--all-checkpoints requires --run-name.")
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        epoch_checkpoints = sorted(ckpt_dir.glob("model_epoch_*.pt"))
        if not epoch_checkpoints:
            raise FileNotFoundError(f"No model_epoch_*.pt files found in {ckpt_dir}")
        checkpoint_jobs.extend((path, False) for path in epoch_checkpoints)
        best_path = ckpt_dir / "model_best.pt"
        if best_path.exists():
            checkpoint_jobs.append((best_path, True))
    else:
        if args.checkpoint is not None:
            ckpt_path = Path(args.checkpoint).expanduser().resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            checkpoint_jobs.append((ckpt_path, ckpt_path.name == "model_best.pt"))
            if run_dir is None and ckpt_path.parent.name == "checkpoints":
                run_dir = ckpt_path.parent.parent
        else:
            if run_dir is None:
                raise ValueError("Either --run-name or --checkpoint must be provided.")
            ckpt_dir = run_dir / "checkpoints"
            ckpt_path = ckpt_dir / "model_best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Pass --checkpoint explicitly.")
            checkpoint_jobs.append((ckpt_path, True))

    if not checkpoint_jobs and not use_ensemble:
        raise RuntimeError("No checkpoints selected for evaluation.")

    if max_checkpoints is not None and len(checkpoint_jobs) > max_checkpoints:
        anchor_indices = {0, len(checkpoint_jobs) - 1}
        best_index = next((idx for idx, (_, is_best) in enumerate(checkpoint_jobs) if is_best), None)
        if best_index is not None:
            anchor_indices.add(best_index)
        required = len(anchor_indices)
        if max_checkpoints < required:
            print(
                f"[Eval] Requested --max-checkpoints={max_checkpoints} is smaller than required anchors "
                f"({required}); using {required} instead."
            )
            max_checkpoints = required

        total = len(checkpoint_jobs)
        selected = set(anchor_indices)
        remaining_slots = max_checkpoints - len(selected)
        if remaining_slots > 0 and total > len(selected):
            span = total - 1
            for i in range(1, remaining_slots + 1):
                raw_idx = int(round(i * span / (remaining_slots + 1)))
                raw_idx = max(1, min(total - 2, raw_idx))
                idx = raw_idx
                direction = 1
                while idx in selected:
                    idx = raw_idx + direction
                    if idx >= total - 1:
                        idx = raw_idx - direction
                        direction += 1
                    else:
                        direction += 1
                    if idx <= 0 or idx >= total - 1:
                        idx = raw_idx
                        break
                if idx not in selected and 0 < idx < total - 1:
                    selected.add(idx)

        selected_indices = sorted(selected)
        checkpoint_jobs = [checkpoint_jobs[idx] for idx in selected_indices]
        print(
            f"[Eval] Limiting evaluation to {len(checkpoint_jobs)} checkpoints "
            f"({', '.join(path.name for path, _ in checkpoint_jobs)})"
        )

    metrics_path: Path | None = None
    metrics_dir: Path | None = None
    if args.metrics_file is not None:
        metrics_path = Path(args.metrics_file).expanduser().resolve()
        metrics_dir = metrics_path.parent
    elif run_dir is not None:
        metrics_dir = run_dir / "metrics"
        metrics_path = metrics_dir / "metrics.json"

    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if metrics_dir.name == "metrics" and metrics_path is not None:
            legacy_metrics = metrics_dir.parent / "metrics.json"
            if legacy_metrics.exists() and not metrics_path.exists():
                try:
                    legacy_metrics.replace(metrics_path)
                    print(f"[Eval] Migrated legacy metrics.json into {metrics_dir}")
                except OSError as exc:
                    print(f"[Eval] Warning: failed to migrate legacy metrics.json ({exc}).")

    loss_fn = CompositeSpectralLoss(
        LossWeights(
            lambda_l1=cfg.lambda_l1,
            lambda_sam=cfg.lambda_sam,
            lambda_sid=cfg.lambda_sid,
            lambda_ergas=cfg.lambda_ergas,
            lambda_srgb_l1=cfg.lambda_srgb_l1,
            lambda_srgb_ssim=cfg.lambda_srgb_ssim,
        )
    )
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
    print(
        f"[Eval] Evaluating {len(dataset)} samples | resize={resize_info} | cache={cache_info} ({cache_mode})"
    )
    print(
        f"[Eval] Loss weights: "
        f"lambda_l1={cfg.lambda_l1:.3f} | "
        f"lambda_sam={cfg.lambda_sam:.3f} | "
        f"lambda_sid={cfg.lambda_sid:.3f} | "
        f"lambda_ergas={cfg.lambda_ergas:.3f} | "
        f"lambda_srgb_l1={cfg.lambda_srgb_l1:.3f} | "
        f"lambda_srgb_ssim={cfg.lambda_srgb_ssim:.3f}"
    )
    if use_ensemble:
        print(f"[Eval] Ensemble mode: {len(ensemble_paths)} checkpoints")
    else:
        print(f"[Eval] Checkpoints queued: {len(checkpoint_jobs)}")

    # Resolve TTA mode
    tta_mode = getattr(args, "tta_mode", "none") or "none"
    tta_include_rot90 = getattr(args, "tta_include_rot90", False)
    resolved_tta_mode, tta_transforms = resolve_tta_mode(tta_mode, tta_include_rot90)
    num_tta_passes = len(tta_transforms)
    print(f"[Eval] TTA mode: {resolved_tta_mode} ({num_tta_passes} forward pass{'es' if num_tta_passes > 1 else ''})")

    evaluation_results: List[Tuple[EvaluationAccumulator, Dict[str, float]]] = []
    
    if use_ensemble:
        # Ensemble evaluation: load all models, wrap in EnsembleModel, evaluate once
        ensemble_model, ensemble_info = load_ensemble_models(ensemble_paths, cfg, device)
        
        # Create label for ensemble
        epochs = ensemble_info.get("epoch", [])
        epoch_strs = [str(e) if e is not None else "?" for e in epochs]
        label = f"ensemble-{len(ensemble_paths)}x (epochs: {', '.join(epoch_strs)})"
        
        # Use first checkpoint path for metrics logging reference
        first_ckpt_path = ensemble_paths[0]
        
        accumulators = [
            EvaluationAccumulator(
                label=label,
                epoch=None,  # Ensemble doesn't have a single epoch
                checkpoint_path=first_ckpt_path,
                model=ensemble_model,
                metric_names=metrics_requested,
                is_best=False,
            )
        ]
        
        run_multi_evaluation(
            accumulators,
            dataset,
            loader_kwargs,
            device,
            loss_fn,
            metrics_requested,
            progress_updates=args.progress_updates,
            inference_resize=inference_resize,
            upsample_metrics=upsample_metrics,
            tta_transforms=tta_transforms,
        )
        for accumulator in accumulators:
            evaluation_results.append((accumulator, accumulator.finalize()))
    else:
        # Standard checkpoint evaluation
        parallel_limit = args.max_parallel_checkpoints or len(checkpoint_jobs)
        parallel_limit = max(1, min(parallel_limit, len(checkpoint_jobs)))
        total_chunks = (len(checkpoint_jobs) + parallel_limit - 1) // parallel_limit

        for chunk_index, chunk in enumerate(chunked(checkpoint_jobs, parallel_limit), start=1):
            if total_chunks > 1:
                print(
                    f"[Eval] Processing checkpoint chunk {chunk_index}/{total_chunks} "
                    f"({len(chunk)} checkpoint(s))"
                )

            accumulators: List[EvaluationAccumulator] = []
            for ckpt_path, is_best in chunk:
                model_cfg = copy.deepcopy(cfg)
                model, info = load_model(ckpt_path, model_cfg, device)
                epoch_from_state = info.get("epoch")
                epoch = epoch_from_state if epoch_from_state is not None else infer_epoch_from_path(ckpt_path)
                label = f"epoch-{epoch:03d}" if epoch is not None else ckpt_path.stem
                if is_best:
                    label = f"best (ep {epoch})" if epoch is not None else "best"
                num_params = sum(p.numel() for p in model.parameters())
                print(
                    f"[Eval] Loaded {ckpt_path.name}: {num_params / 1e6:.2f}M params | label={label}"
                )
                accumulators.append(
                    EvaluationAccumulator(
                        label=label,
                        epoch=epoch,
                        checkpoint_path=ckpt_path,
                        model=model,
                        metric_names=metrics_requested,
                        is_best=is_best,
                    )
                )

            run_multi_evaluation(
                accumulators,
                dataset,
                loader_kwargs,
                device,
                loss_fn,
                metrics_requested,
                progress_updates=args.progress_updates,
                inference_resize=inference_resize,
                upsample_metrics=upsample_metrics,
                tta_transforms=tta_transforms,
            )
            for accumulator in accumulators:
                evaluation_results.append((accumulator, accumulator.finalize()))

    summary_order = ["loss"] + [name for name in metrics_requested if name != "loss"]
    print("[Eval] Completed evaluation.")
    for accumulator, results in evaluation_results:
        epoch_display = accumulator.epoch if accumulator.epoch is not None else "?"
        summary = " | ".join(
            f"{metric}={results[metric]:.4f}"
            for metric in summary_order
            if metric in results
        )
        print(f"[Eval] Summary :: {accumulator.label} (epoch {epoch_display}) | {summary}")

    if metrics_path is None:
        print("[Eval] Metrics log path not provided; skipping JSON update.")
    else:
        try:
            history = load_metrics_history(metrics_path)
        except ValueError as exc:
            print(f"[Eval] Warning: {exc}. Overwriting metrics history ({exc}).")
            history = []

        section_key = "val_native" if cfg.resize_to is None else f"eval_{resolution_tag(cfg.resize_to)}"
        manual_epoch_override = (
            args.epoch if len(evaluation_results) == 1 and args.epoch is not None else None
        )
        results_written = 0
        for accumulator, results in evaluation_results:
            epoch_for_history = (
                manual_epoch_override if manual_epoch_override is not None else accumulator.epoch
            )
            if epoch_for_history is None:
                print(
                    f"[Eval] Warning: unable to determine epoch for {accumulator.checkpoint_path.name}; "
                    "skipping history update."
                )
                continue

            record = ensure_epoch_record(history, int(epoch_for_history))
            metrics_payload = {key: float(value) for key, value in results.items()}
            section_units = units_from_defaults(metrics_payload.keys(), ACCURACY_METRIC_UNITS)
            section_context = {
                "source": SOURCE_EVALUATE,
                "resolution": resolution_label(cfg.resize_to),
                "checkpoint": accumulator.checkpoint_path.name,
            }
            record[section_key] = metrics_payload
            record.setdefault("units", {})[section_key] = dict(section_units)
            record.setdefault("context", {})[section_key] = section_context
            record.setdefault("updated", {})[section_key] = utc_timestamp()
            results_written += 1

        if results_written:
            save_metrics_history(metrics_path, history)
            print(
                f"[Eval] Logged metrics for {results_written} checkpoint(s) to {metrics_path}."
            )
        else:
            print("[Eval] No metrics written (missing epoch metadata).")

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Track 1 checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt). If omitted, uses --run-name/model_best.pt")
    parser.add_argument(
        "--ensemble-checkpoints",
        type=str,
        default=None,
        help=(
            "Comma-separated paths to checkpoints for ensemble inference. "
            "When set, predictions are averaged across all models. "
            "All checkpoints must be the same architecture variant."
        ),
    )
    parser.add_argument(
        "--fix-best-checkpoint",
        action="store_true",
        help="Check metrics.json and update model_best.pt if it doesn't match the actual best epoch.",
    )
    parser.add_argument(
        "--fix-best-dry-run",
        action="store_true",
        help="Like --fix-best-checkpoint but only reports what would be done.",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Evaluate every checkpoint under the run directory (requires --run-name).",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=0,
        help="Maximum number of checkpoints to evaluate (first/last/best are always included).",
    )
    parser.add_argument(
        "--max-parallel-checkpoints",
        type=int,
        default=0,
        help="Limit how many checkpoints are loaded simultaneously (0 means all).",
    )
    parser.add_argument(
        "--upsample-metrics",
        action="store_true",
        help="Run the model at the training resolution but upsample predictions back to native size before computing metrics.",
    )
    parser.add_argument(
        "--tta-mode",
        type=str,
        default="none",
        choices=("none", "flip", "rotate90", "dihedral", "auto"),
        help=(
            "Test-time augmentation mode. "
            "none=disabled (1 pass), "
            "flip=H/V flips (4 passes), "
            "rotate90=90° rotations (4 passes), "
            "dihedral=flips+rotations (8 passes), "
            "auto=flip by default, dihedral if --tta-include-rot90."
        ),
    )
    parser.add_argument(
        "--tta-include-rot90",
        action="store_true",
        help="When --tta-mode=auto, upgrade from flip to dihedral (matches --aug-rotate90 training).",
    )
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
        "--lambda-ergas",
        type=float,
        default=None,
        help="Weight for ERGAS loss term (default 0.0).",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        help="Model architecture variant to evaluate (baseline, unet_lite).",
    )
    parser.add_argument(
        "--unet-base-channels",
        type=int,
        default=None,
        help="Base channel count for UNet-lite variant (default 32).",
    )
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=None,
        help="Latent channel width inside the UNet-lite decoder head (default 32).",
    )
    parser.add_argument(
        "--encoder-depth",
        type=int,
        default=None,
        help="Number of stride-2 encoder stages for UNet-lite (default 3).",
    )
    parser.add_argument(
        "--coarse-channels",
        type=int,
        default=None,
        help="Number of coarse spectral channels before interpolation (default from config).",
    )
    parser.add_argument(
        "--norm-type",
        type=str,
        default=None,
        choices=("group", "rms", "none"),
        help="Normalization applied inside UNet-lite blocks (group, rms, none).",
    )
    parser.add_argument(
        "--use-residual-head",
        action="store_true",
        help="Enable coarse+residual refinement head (must match training).",
    )
    parser.add_argument(
        "--use-spectral-conv",
        action="store_true",
        help="Enable 1D spectral convolution after the spectral head (must match training).",
    )
    parser.add_argument(
        "--spectral-conv-kernel-size",
        type=int,
        default=None,
        help="Kernel size for spectral 1D convolution (odd, e.g., 3 or 5).",
    )
    parser.add_argument(
        "--decoder-dropout",
        type=float,
        default=None,
        help="Dropout probability for decoder/bottleneck residual blocks (must match training).",
    )
    parser.add_argument(
        "--stochastic-depth-p",
        type=float,
        default=None,
        help="Stochastic depth drop probability for decoder/bottleneck blocks (must match training).",
    )
    parser.add_argument(
        "--use-bottleneck-attention",
        action="store_true",
        help="Enable compact attention in the bottleneck (must match training).",
    )
    parser.add_argument(
        "--use-raw-input-skip",
        action="store_true",
        help="MST++: Enable raw input skip connection (must match training).",
    )
    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=None,
        help="Kernel size for UNet-lite residual blocks (default 3; must match training).",
    )
    parser.add_argument(
        "--resize-to",
        type=int,
        default=0,
        help="Optional spatial size to resize inputs/targets to (e.g., 64). Use 0 to keep native size.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        choices=("native", "resized"),
        help="Convenience flag for switching between native evaluation and the configured resize.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the run under src/models/simple_cnn/runs/ (used when --checkpoint is omitted).",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Path to the metrics JSON file to update. Defaults to <run-dir>/metrics.json when available.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number to associate logged metrics with (falls back to checkpoint metadata).",
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
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=None,
        help="Weight for L1 loss term (default 1.0).",
    )
    parser.add_argument(
        "--lambda-sam",
        type=float,
        default=None,
        help="Weight for SAM loss term (default 0.1).",
    )
    parser.add_argument(
        "--lambda-sid",
        type=float,
        default=None,
        help="Weight for SID loss term (default 0.0).",
    )
    parser.add_argument(
        "--lambda-srgb-l1",
        type=float,
        default=None,
        help="Weight for sRGB linear L1 loss term (default 0.0).",
    )
    parser.add_argument(
        "--lambda-srgb-ssim",
        type=float,
        default=None,
        help="Weight for sRGB SSIM perceptual loss term (default 0.0).",
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)


