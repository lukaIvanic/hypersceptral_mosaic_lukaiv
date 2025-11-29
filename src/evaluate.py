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
) -> None:
    if not accumulators:
        return

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
            preds = run_model_with_resize(
                accumulator.model,
                inputs,
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

    if not checkpoint_jobs:
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
    print(f"[Eval] Checkpoints queued: {len(checkpoint_jobs)}")

    parallel_limit = args.max_parallel_checkpoints or len(checkpoint_jobs)
    parallel_limit = max(1, min(parallel_limit, len(checkpoint_jobs)))
    total_chunks = (len(checkpoint_jobs) + parallel_limit - 1) // parallel_limit

    evaluation_results: List[Tuple[EvaluationAccumulator, Dict[str, float]]] = []
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


