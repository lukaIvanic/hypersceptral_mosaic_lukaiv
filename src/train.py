from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import math

from .config import TrainConfig
from .data import create_dataloaders
from .losses import CompositeSpectralLoss, LossWeights
from .metrics import aggregate
from .models.builder import create_model
from .utils.inference import run_model_with_resize, sliding_window_inference
from .utils.metrics.schema import (
    ACCURACY_METRIC_UNITS,
    SOURCE_TRAIN,
    TRAINING_METRIC_UNITS,
    resolution_label,
    resolution_tag,
    units_from_defaults,
)
from .utils.metrics.storage import (
    load_metrics_history,
    remove_epoch_record,
    save_metrics_history,
    utc_timestamp,
)

try:
    import psutil  # type: ignore[import-error]
except ImportError:  # pragma: no cover
    psutil = None

logger = logging.getLogger(__name__)
_PROCESS = psutil.Process() if psutil is not None else None


def _log_partial_load(result: Any, context: str) -> None:
    if result is None:
        return
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    if missing:
        logger.warning("[%s] Missing keys when loading state_dict: %s", context, missing)
    if unexpected:
        logger.warning("[%s] Unexpected keys when loading state_dict: %s", context, unexpected)


def _time_block(
    device: torch.device,
    fn: Callable[[], Any],
    sync_cuda: bool,
) -> tuple[Any, float]:
    """
    Measure the wall-time of ``fn`` with CUDA-aware synchronization.
    Returns (result, seconds).
    """

    if device.type == "cuda" and sync_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = fn()
        end_event.record()
        torch.cuda.synchronize(device)
        duration = start_event.elapsed_time(end_event) / 1000.0
        return result, duration

    start = time.perf_counter()
    result = fn()
    duration = time.perf_counter() - start
    return result, duration


class WarmupCosineScheduler:
    """
    Lightweight cosine scheduler with optional linear warmup.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        total_epochs: int,
        warmup_epochs: int,
        warmup_start_factor: float,
        eta_min: float,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_epochs = max(total_epochs, 1)
        self.warmup_epochs = max(min(warmup_epochs, self.total_epochs - 1), 0)
        self.warmup_start_factor = float(max(min(warmup_start_factor, 1.0), 1e-4))
        self.eta_min = float(max(min(eta_min, base_lr), 0.0))
        self._last_lr = base_lr

    def _compute_warmup_lr(self, epoch_index: int) -> float:
        if self.warmup_epochs <= 0:
            return self.base_lr
        if self.warmup_epochs == 1:
            return self.base_lr
        progress = epoch_index / max(self.warmup_epochs - 1, 1)
        progress = float(max(min(progress, 1.0), 0.0))
        factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
        return self.base_lr * factor

    def _compute_cosine_lr(self, epoch_index: int) -> float:
        cosine_epochs = max(self.total_epochs - self.warmup_epochs, 1)
        progress = (epoch_index - self.warmup_epochs) / cosine_epochs
        progress = float(max(min(progress, 1.0), 0.0))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        if self.base_lr <= 0:
            return self.eta_min
        min_factor = self.eta_min / self.base_lr
        factor = min_factor + (1.0 - min_factor) * cosine
        return self.base_lr * factor

    def step(self, epoch_index: int) -> float:
        if epoch_index < self.warmup_epochs:
            lr = self._compute_warmup_lr(epoch_index)
        else:
            lr = self._compute_cosine_lr(epoch_index)

        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._last_lr = lr
        return lr

    def get_last_lr(self) -> float:
        return self._last_lr


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> Optional[WarmupCosineScheduler]:
    name = getattr(cfg, "lr_scheduler", "none").lower()
    if name in {"none", "", "off"}:
        return None
    if name != "cosine":
        raise ValueError(
            f"Unknown scheduler '{cfg.lr_scheduler}'. Supported options: none, cosine."
        )

    base_lr = optimizer.param_groups[0]["lr"]
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        base_lr=base_lr,
        total_epochs=cfg.epochs,
        warmup_epochs=cfg.scheduler_warmup_epochs,
        warmup_start_factor=cfg.scheduler_warmup_start_factor,
        eta_min=cfg.scheduler_min_lr,
    )
    logger.info(
        "[Scheduler] cosine | warmup_epochs=%d | start_factor=%.3f | eta_min=%.2e",
        scheduler.warmup_epochs,
        cfg.scheduler_warmup_start_factor,
        cfg.scheduler_min_lr,
    )
    return scheduler


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
    inference_resize: int | None,
    timing_sync: bool,
    grad_accum_steps: int,
    profile_steps: int = 0,
    profile_output_dir: Path | None = None,
    profile_start_step: int = 1,
    profile_with_stack: bool = False,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    num_samples = 0
    steps_total = len(loader)
    grad_accum_steps = max(1, int(grad_accum_steps))
    accum_remainder = steps_total % grad_accum_steps if grad_accum_steps > 0 else 0
    last_group_start = (
        steps_total - accum_remainder + 1 if accum_remainder != 0 else steps_total + 1
    )
    timings = {
        "io": 0.0,
        "preprocess": 0.0,
        "forward": 0.0,
        "loss_fn": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
        "loss_item": 0.0,
        "interp": 0.0,
        "step_wall": 0.0,
        "ram_mb": 0.0,
    }
    prev_time = time.perf_counter()
    profiler = None
    remaining_profile_steps = max(0, profile_steps)
    profile_ready = remaining_profile_steps > 0
    trace_handler = None
    torch_profile = None
    ProfilerActivity = None
    if profile_ready:
        try:
            from torch.profiler import ProfilerActivity as _ProfilerActivity
            from torch.profiler import profile as _torch_profile
            from torch.profiler import tensorboard_trace_handler
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - depends on torch build
            logger.warning("[Profiler] torch.profiler unavailable (%s); disabling profiling.", exc)
            profile_ready = False
        else:
            ProfilerActivity = _ProfilerActivity
            torch_profile = _torch_profile
            if profile_output_dir is not None:
                profile_output_dir.mkdir(parents=True, exist_ok=True)
                trace_handler = tensorboard_trace_handler(str(profile_output_dir))
            else:
                trace_handler = None

    def _maybe_start_profiler(current_step: int) -> None:
        nonlocal profiler, profile_ready
        if (
            not profile_ready
            or profiler is not None
            or remaining_profile_steps <= 0
            or current_step < max(1, profile_start_step)
        ):
            return
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        profiler = torch_profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=profile_with_stack,
            with_flops=False,
            on_trace_ready=trace_handler,
        )
        profiler.__enter__()
        logger.info(
            "[Profiler] Capturing %d training step(s) starting at step %d of epoch %d%s",
            remaining_profile_steps,
            current_step,
            epoch,
            "" if profile_output_dir is None else f" (traces -> {profile_output_dir})",
        )

    optimizer.zero_grad(set_to_none=True)

    try:
        for step, batch in enumerate(loader, start=1):
            step_start_time = prev_time
            _maybe_start_profiler(step)
            io_time = time.perf_counter() - prev_time

            def _move_to_device() -> tuple[torch.Tensor, torch.Tensor]:
                return (
                    batch["input"].to(device, non_blocking=True),
                    batch["target"].to(device, non_blocking=True),
                )

            (inputs, targets), preprocess_time = _time_block(device, _move_to_device, timing_sync)
            batch_size = inputs.size(0)

            final_shape = tuple(int(dim) for dim in targets.shape[-2:])
            preds, forward_time = _time_block(
                device,
                lambda: run_model_with_resize(model, inputs, inference_resize, final_shape),
                timing_sync,
            )
            interp_time = getattr(model, "last_interp_time", 0.0)

            loss_raw, loss_fn_time = _time_block(device, lambda: loss_fn(preds, targets), timing_sync)
            current_accum_target = (
                accum_remainder if accum_remainder != 0 and step >= last_group_start else grad_accum_steps
            )
            loss = loss_raw / current_accum_target
            _, backward_time = _time_block(device, lambda: loss.backward(), timing_sync)
            accum_position = ((step - 1) % grad_accum_steps) + 1
            if accum_position > current_accum_target:
                accum_position = current_accum_target
            optimizer_time = 0.0
            should_step = (step % grad_accum_steps == 0) or (step == steps_total)
            if should_step:
                _, optimizer_time = _time_block(device, optimizer.step, timing_sync)
                optimizer.zero_grad(set_to_none=True)
            loss_item_start = time.perf_counter()
            loss_value = float(loss_raw.detach().item())
            loss_item_time = time.perf_counter() - loss_item_start

            running_loss += loss_value * batch_size
            num_samples += batch_size

            timings["io"] += io_time
            timings["preprocess"] += preprocess_time
            timings["forward"] += forward_time
            timings["loss_fn"] += loss_fn_time
            timings["backward"] += backward_time
            timings["optimizer"] += optimizer_time
            timings["loss_item"] += loss_item_time
            timings["interp"] += interp_time
            if _PROCESS is not None:
                ram_mb = _PROCESS.memory_info().rss / (1024 ** 2)
                timings["ram_mb"] += ram_mb
            else:
                ram_mb = None

            step_end_time = time.perf_counter()
            timings["step_wall"] += step_end_time - step_start_time
            prev_time = step_end_time

            logger.debug(
                "epoch %d step %d/%d | io=%.2f ms | preprocess=%.2f ms | forward=%.2f ms | "
                "loss_fn=%.2f ms | backward=%.2f ms | optimizer=%.2f ms | loss_item=%.2f ms | "
                "interp=%.2f ms | wall=%.2f ms | accum=%d/%d%s",
                epoch,
                step,
                steps_total,
                io_time * 1e3,
                preprocess_time * 1e3,
                forward_time * 1e3,
                loss_fn_time * 1e3,
                backward_time * 1e3,
                optimizer_time * 1e3,
                loss_item_time * 1e3,
                interp_time * 1e3,
                (step_end_time - step_start_time) * 1e3,
                accum_position,
                current_accum_target,
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

            if profiler is not None:
                profiler.step()
                remaining_profile_steps -= 1
                if remaining_profile_steps <= 0:
                    profiler.__exit__(None, None, None)
                    profiler = None
                    profile_ready = False
                    logger.info("[Profiler] Completed requested capture for epoch %d.", epoch)
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
            profile_ready = False

    avg_times = {k: (v / max(steps_total, 1)) for k, v in timings.items()}
    avg_times["loss"] = running_loss / max(num_samples, 1)
    return avg_times


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    inference_resize: int | None,
    val_crop_size: int | None,
    val_crop_overlap: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    counts = 0
    metrics_sum = {k: 0.0 for k in ("mae", "mse", "psnr", "sam", "ergas")}

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        batch_size = inputs.size(0)

        final_shape = tuple(int(dim) for dim in targets.shape[-2:])
        if val_crop_size is not None:
            preds_list = []
            for index in range(batch_size):
                pred_tensor = sliding_window_inference(
                    model,
                    inputs[index],
                    val_crop_size,
                    val_crop_overlap,
                    inference_resize,
                )
                preds_list.append(pred_tensor)
            preds = torch.stack(preds_list, dim=0)
        else:
            preds = run_model_with_resize(model, inputs, inference_resize, final_shape)
        loss = loss_fn(preds, targets)

        total_loss += loss.item() * batch_size
        counts += batch_size

        batch_metrics = aggregate(preds, targets, metrics=("mae", "mse", "psnr", "sam", "ergas"))
        for key, value in batch_metrics.items():
            metrics_sum[key] += value * batch_size

    results = {k: v / max(counts, 1) for k, v in metrics_sum.items()}
    results["loss"] = total_loss / max(counts, 1)
    return results


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    model_variant: str | None = None,
) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "variant": model_variant or getattr(model, "variant_name", None),
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
    if getattr(args, "step_batch_size", None) is not None:
        cfg.train_micro_batch_size = max(1, int(args.step_batch_size))
    if getattr(args, "accum_batch_size", None) is not None:
        cfg.effective_batch_size = max(1, int(args.accum_batch_size))
    if getattr(args, "train_crop_size", None) is not None:
        crop_val = int(args.train_crop_size)
        if crop_val <= 0:
            raise ValueError("--train-crop-size must be a positive integer.")
        cfg.train_crop_size = crop_val
    if getattr(args, "val_crop_size", None) is not None:
        crop_val = int(args.val_crop_size)
        if crop_val <= 0:
            raise ValueError("--val-crop-size must be a positive integer.")
        cfg.val_crop_size = crop_val
    if getattr(args, "val_crop_overlap", None) is not None:
        cfg.val_crop_overlap = float(args.val_crop_overlap)
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
    if getattr(args, "ram_cache", False):
        cfg.ram_cache = True
    if getattr(args, "resume", False):
        cfg.resume = True
    if args.device is not None:
        cfg.device = args.device
    if args.hidden_channels is not None:
        cfg.hidden_channels = args.hidden_channels
    if args.log_interval is not None:
        cfg.log_interval = args.log_interval
    if args.resize_to is not None:
        cfg.resize_to = args.resize_to if args.resize_to > 0 else None
    if getattr(args, "train_inference_resize", None) is not None:
        value = args.train_inference_resize
        cfg.train_inference_resize = value if value and value > 0 else None
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
    if getattr(args, "lr_scheduler", None) is not None:
        cfg.lr_scheduler = args.lr_scheduler.lower()
    if getattr(args, "scheduler_warmup_epochs", None) is not None:
        cfg.scheduler_warmup_epochs = args.scheduler_warmup_epochs
    if getattr(args, "scheduler_warmup_start_factor", None) is not None:
        cfg.scheduler_warmup_start_factor = args.scheduler_warmup_start_factor
    if getattr(args, "scheduler_min_lr", None) is not None:
        cfg.scheduler_min_lr = args.scheduler_min_lr

    # Profiling options
    if getattr(args, "profile_steps", None) is not None:
        cfg.profile_steps = max(0, int(args.profile_steps))
    if getattr(args, "profile_epoch", None) is not None:
        cfg.profile_epoch = max(1, int(args.profile_epoch))
    if getattr(args, "profile_dir", None) is not None:
        value = args.profile_dir.strip()
        cfg.profile_dir = None if value == "" else Path(value)
    if getattr(args, "profile_start_step", None) is not None:
        cfg.profile_start_step = max(1, int(args.profile_start_step))
    if getattr(args, "profile_with_stack", False):
        cfg.profile_with_stack = True
    if getattr(args, "use_compile", False):
        cfg.use_compile = True
    if getattr(args, "no_timing_sync", False):
        cfg.timing_sync = False

    # Data augmentation flags
    if getattr(args, "aug_rotate90", False):
        cfg.aug_rotate90 = True
    if getattr(args, "aug_resized_crop", False):
        cfg.aug_resized_crop = True
    if getattr(args, "aug_intensity_jitter", False):
        cfg.aug_intensity_jitter = True

    # Advanced model options
    if getattr(args, "use_residual_head", False):
        cfg.use_residual_head = True
    if getattr(args, "use_spectral_conv", False):
        cfg.use_spectral_conv = True
    if getattr(args, "spectral_conv_kernel_size", None) is not None:
        cfg.spectral_conv_kernel_size = max(1, int(args.spectral_conv_kernel_size))
    if getattr(args, "use_bottleneck_attention", False):
        cfg.use_bottleneck_attention = True
    if getattr(args, "conv_kernel_size", None) is not None:
        cfg.conv_kernel_size = max(1, int(args.conv_kernel_size))
    if getattr(args, "norm_type", None) is not None:
        cfg.norm_type = args.norm_type.lower()

    # Regularization options
    if getattr(args, "decoder_dropout", None) is not None:
        cfg.decoder_dropout = max(0.0, float(args.decoder_dropout))
    if getattr(args, "stochastic_depth_p", None) is not None:
        cfg.stochastic_depth_p = max(0.0, float(args.stochastic_depth_p))

    if getattr(args, "resume", False) and getattr(args, "init_from", None):
        raise ValueError("Cannot use --resume and --init-from together.")

    step_batch_size = cfg.train_micro_batch_size or cfg.batch_size
    if step_batch_size is None or step_batch_size <= 0:
        raise ValueError("Per-step batch size must be a positive integer.")
    step_batch_size = int(step_batch_size)
    cfg.train_micro_batch_size = step_batch_size
    cfg.batch_size = step_batch_size

    accum_batch_size = cfg.effective_batch_size
    if accum_batch_size is not None:
        accum_batch_size = int(accum_batch_size)
        if accum_batch_size < step_batch_size:
            raise ValueError(
                "Accumulated batch size must be greater than or equal to the per-step batch size."
            )
        if accum_batch_size % step_batch_size != 0:
            raise ValueError(
                "Accumulated batch size must be an integer multiple of the per-step batch size."
            )
        cfg.grad_accumulation_steps = accum_batch_size // step_batch_size
    else:
        cfg.grad_accumulation_steps = max(1, int(cfg.grad_accumulation_steps or 1))
        accum_batch_size = step_batch_size * cfg.grad_accumulation_steps
        cfg.effective_batch_size = accum_batch_size

    if cfg.grad_accumulation_steps < 1:
        raise ValueError("Gradient accumulation steps must be at least 1.")

    if cfg.train_crop_size is not None:
        allowed_resize = {None, 1024}
        if cfg.resize_to not in allowed_resize:
            raise ValueError(
                "--train-crop-size requires --resize-to to be unset or 1024."
            )
        if cfg.train_crop_size > 1024:
            raise ValueError("train_crop_size must be <= 1024 for the native dataset.")

    if cfg.val_crop_size is None and cfg.train_crop_size is not None:
        cfg.val_crop_size = cfg.train_crop_size

    if cfg.val_crop_size is not None:
        allowed_resize = {None, 1024}
        if cfg.resize_to not in allowed_resize:
            raise ValueError(
                "--val-crop-size requires --resize-to to be unset or 1024."
            )
        if cfg.val_crop_size > 1024:
            raise ValueError("val_crop_size must be <= 1024 for the native dataset.")

    if cfg.val_crop_overlap < 0.0 or cfg.val_crop_overlap >= 1.0:
        raise ValueError("--val-crop-overlap must be in [0.0, 1.0).")
    if cfg.val_crop_overlap > 0.0 and cfg.val_crop_size is None:
        raise ValueError("--val-crop-overlap requires --val-crop-size to be set.")

    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    model_runs_root = Path(__file__).resolve().parent / "models" / "simple_cnn" / "runs"
    run_dir = model_runs_root / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if cfg.profile_steps > 0 and cfg.profile_dir is None:
        cfg.profile_dir = run_dir / "profiling"

    history_path = metrics_dir / "metrics.json"
    legacy_history_path = run_dir / "metrics.json"
    if legacy_history_path.exists() and not history_path.exists():
        try:
            legacy_history_path.replace(history_path)
            logger.info("[Metrics] Migrated legacy metrics.json into %s", metrics_dir)
        except OSError as exc:
            logger.warning(
                "[Metrics] Failed to move legacy metrics.json (%s); will read inline.", exc
            )
            history_path = legacy_history_path
    try:
        history: list[Dict[str, Any]] = load_metrics_history(history_path)
    except ValueError as exc:
        logger.warning("%s; starting a new metrics log.", exc)
        history = []

    history_best_val = float("inf")
    last_logged_epoch = 0
    for record in history:
        epoch_value = record.get("epoch")
        if isinstance(epoch_value, (int, float)):
            last_logged_epoch = max(last_logged_epoch, int(epoch_value))
        val_metrics = record.get("val")
        if isinstance(val_metrics, dict) and "loss" in val_metrics:
            try:
                loss_value = float(val_metrics["loss"])
            except (TypeError, ValueError):
                continue
            if loss_value < history_best_val:
                history_best_val = loss_value

    resize_info = cfg.resize_to if cfg.resize_to is not None else "native"
    cache_info = cfg.cache_dir if cfg.cache_dir is not None else "disabled"
    cache_mode = "rw" if cfg.write_cache else "ro"
    ram_cache_info = "on" if cfg.ram_cache else "off"
    resume_info = "on" if cfg.resume else "off"
    logger.info(
        "[Setup] device=%s | data_root=%s | resize=%s | cache=%s (%s) | ram_cache=%s | resume=%s | variant=%s | run_dir=%s",
        device,
        cfg.data_root,
        resize_info,
        cache_info,
        cache_mode,
        ram_cache_info,
        resume_info,
        cfg.model_variant,
        run_dir,
    )
    logger.info(
        "[Batch] step=%d | accum_steps=%d | effective=%d",
        cfg.train_micro_batch_size,
        cfg.grad_accumulation_steps,
        cfg.effective_batch_size,
    )
    if cfg.train_crop_size is not None:
        logger.info("[Patch] train_crop_size=%d (random training crops enabled)", cfg.train_crop_size)
    if cfg.val_crop_size is not None:
        logger.info(
            "[ValPatch] val_crop_size=%d | overlap=%.2f (sliding-window validation)",
            cfg.val_crop_size,
            cfg.val_crop_overlap,
        )
    logger.info(
        "[Loss] lambda_l1=%.3f | lambda_sam=%.3f | lambda_sid=%.3f | "
        "lambda_ergas=%.3f | lambda_srgb_l1=%.3f | lambda_srgb_ssim=%.3f",
        cfg.lambda_l1,
        cfg.lambda_sam,
        cfg.lambda_sid,
        cfg.lambda_ergas,
        cfg.lambda_srgb_l1,
        cfg.lambda_srgb_ssim,
    )
    if device.type == "cuda" and not cfg.timing_sync:
        logger.info("[Timing] CUDA synchronization disabled; per-step timings are approximate.")
    if _PROCESS is None:
        logger.warning("psutil not available; RAM telemetry disabled.")

    train_loader, val_loader = create_dataloaders(cfg)

    model = create_model(
        cfg.model_variant,
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
    ).to(device)
    if cfg.use_compile:
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("[torch.compile] Enabled (mode=reduce-overhead).")
            except Exception as exc:  # pragma: no cover - depends on torch version
                logger.warning("[torch.compile] Failed (%s); continuing without.", exc)
        else:
            logger.warning("[torch.compile] Requested but torch.compile is unavailable in this PyTorch build.")
    allow_partial_load = getattr(args, "allow_partial_load", False)
    strict_checkpoint_load = not allow_partial_load

    init_from = getattr(args, "init_from", None)
    if init_from:
        ckpt_path = Path(init_from)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--init-from checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        load_result = model.load_state_dict(state_dict, strict=strict_checkpoint_load)
        if allow_partial_load:
            _log_partial_load(load_result, "Init")
        logger.info("[Init] Loaded weights from %s", ckpt_path)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "[Model] variant=%s | params=%.2fM",
        getattr(model, "variant_name", cfg.model_variant),
        num_params / 1e6,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = create_scheduler(optimizer, cfg)
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

    best_val = history_best_val if history_best_val < float("inf") else float("inf")
    start_epoch = 1

    if cfg.resume:
        last_ckpt_epoch = 0
        last_ckpt_path: Optional[Path] = None
        for ckpt_path in sorted(ckpt_dir.glob("model_epoch_*.pt")):
            try:
                epoch_num = int(ckpt_path.stem.split("_")[-1])
            except ValueError:
                continue
            if epoch_num > last_ckpt_epoch:
                last_ckpt_epoch = epoch_num
                last_ckpt_path = ckpt_path

        if last_ckpt_path is None:
            logger.warning(
                "[Resume] Requested but no checkpoint found in %s (last logged epoch=%d); starting from scratch.",
                ckpt_dir,
                last_logged_epoch,
            )
        else:
            checkpoint = torch.load(last_ckpt_path, map_location=device)
            load_result = model.load_state_dict(
                checkpoint["model"], strict=strict_checkpoint_load
            )
            if allow_partial_load:
                _log_partial_load(load_result, "Resume")
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None:
                try:
                    optimizer.load_state_dict(optimizer_state)
                except (ValueError, RuntimeError) as exc:
                    if strict_checkpoint_load:
                        raise
                    logger.warning(
                        "[Resume] Optimizer state mismatch (%s); starting with fresh optimizer state.",
                        exc,
                    )
            saved_epoch = int(checkpoint.get("epoch", last_ckpt_epoch))
            start_epoch = saved_epoch + 1
            logger.info(
                "[Resume] Loaded checkpoint %s (epoch=%d); resuming from epoch %d.",
                last_ckpt_path.name,
                saved_epoch,
                start_epoch,
            )

    if best_val < float("inf"):
        logger.info("[Progress] Historical best validation loss: %.4f", best_val)

    if start_epoch > cfg.epochs:
        logger.info(
            "[Resume] Last completed epoch %d >= requested epochs %d; nothing to do.",
            start_epoch - 1,
            cfg.epochs,
        )
        return

    for epoch in range(start_epoch, cfg.epochs + 1):
        if scheduler is not None:
            current_lr = scheduler.step(epoch - 1)
        else:
            current_lr = optimizer.param_groups[0]["lr"]
        epoch_profile_steps = 0
        epoch_profile_dir: Path | None = None
        if cfg.profile_steps > 0 and epoch == cfg.profile_epoch:
            profile_root = cfg.profile_dir or (run_dir / "profiling")
            epoch_profile_dir = profile_root / f"epoch_{epoch:03d}"
            epoch_profile_steps = cfg.profile_steps
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_fn,
            cfg.log_interval,
            epoch,
            cfg.train_inference_resize,
            cfg.timing_sync,
            cfg.grad_accumulation_steps,
            profile_steps=epoch_profile_steps,
            profile_output_dir=epoch_profile_dir,
            profile_start_step=cfg.profile_start_step,
            profile_with_stack=cfg.profile_with_stack,
        )
        train_stats["lr"] = current_lr
        logger.info("[Train] Epoch %d done | loss=%.4f", epoch, train_stats["loss"])
        msg = (
            "Epoch %d timing (ms): io=%.2f | preprocess=%.2f | forward=%.2f | "
            "loss_fn=%.2f | backward=%.2f | optimizer=%.2f | loss_item=%.2f | interp=%.2f | wall=%.2f"
        )
        args = [
            epoch,
            train_stats["io"] * 1e3,
            train_stats["preprocess"] * 1e3,
            train_stats["forward"] * 1e3,
            train_stats["loss_fn"] * 1e3,
            train_stats["backward"] * 1e3,
            train_stats["optimizer"] * 1e3,
            train_stats["loss_item"] * 1e3,
            train_stats["interp"] * 1e3,
            train_stats["step_wall"] * 1e3,
        ]
        if _PROCESS is not None and train_stats.get("ram_mb") is not None:
            msg += " | ram=%.1f MB"
            args.append(train_stats["ram_mb"])
        logger.info(msg, *args)

        train_timestamp = utc_timestamp()
        record: Dict[str, Any] = {
            "epoch": epoch,
            "timestamp": train_timestamp,
            "train": train_stats,
            "units": {
                "train": units_from_defaults(train_stats.keys(), TRAINING_METRIC_UNITS)
            },
            "context": {
                "train": {
                    "source": SOURCE_TRAIN,
                    "resolution": resolution_label(cfg.resize_to),
                }
            },
            "updated": {"train": train_timestamp},
        }

        if epoch % cfg.val_interval == 0:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                loss_fn,
                cfg.train_inference_resize,
                cfg.val_crop_size,
                cfg.val_crop_overlap,
            )
            metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            logger.info("[Val] Epoch %d | %s", epoch, metrics_str)
            val_section = f"val_{resolution_tag(cfg.resize_to)}"
            val_units = units_from_defaults(val_metrics.keys(), ACCURACY_METRIC_UNITS)
            val_timestamp = utc_timestamp()

            record[val_section] = val_metrics
            record["units"][val_section] = val_units
            record["context"][val_section] = {
                "source": SOURCE_TRAIN,
                "resolution": resolution_label(cfg.resize_to),
            }
            record["updated"][val_section] = val_timestamp

            # Backwards compatibility alias
            record["val"] = val_metrics
            record["units"]["val"] = dict(val_units)
            record["context"]["val"] = dict(record["context"][val_section])
            record["updated"]["val"] = val_timestamp

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    ckpt_dir / "model_best.pt",
                    cfg.model_variant,
                )

        if epoch % cfg.checkpoint_every == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                ckpt_dir / f"model_epoch_{epoch:03d}.pt",
                cfg.model_variant,
            )

        # Update metrics history log
        remove_epoch_record(history, epoch)
        history.append(record)
        history.sort(key=lambda item: item.get("epoch", 0))
        save_metrics_history(history_path, history)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Simple Track 1 model")
    parser.add_argument("--data-root", type=str, default=None, help="Path to data/track1 directory")
    parser.add_argument("--run-name", type=str, default=None, help="Name of the run subdirectory under the model's runs folder")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--step-batch-size",
        type=int,
        default=None,
        help="Per-optimizer-step batch size (micro-batch). Overrides --batch-size for training steps if provided.",
    )
    parser.add_argument(
        "--accum-batch-size",
        type=int,
        default=None,
        help="Effective batch size after gradient accumulation. Must be a multiple of the step batch size.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="Number of training steps to capture with torch.profiler (0 disables profiling).",
    )
    parser.add_argument(
        "--profile-epoch",
        type=int,
        default=1,
        help="Epoch index (1-based) in which profiling should run.",
    )
    parser.add_argument(
        "--profile-start-step",
        type=int,
        default=1,
        help="First training step (1-based) within the profiling epoch to capture.",
    )
    parser.add_argument(
        "--profile-with-stack",
        action="store_true",
        help="Include Python stack traces for profiler events (adds overhead).",
    )
    parser.add_argument(
        "--no-timing-sync",
        action="store_true",
        help="Disable CUDA synchronizations in timing measurements (approximate timings, faster).",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Enable torch.compile (requires PyTorch 2.0+).",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Directory to store profiler traces (defaults to run_dir/profiling).",
    )
    parser.add_argument(
        "--resize-to",
        type=int,
        default=None,
        help="Optional spatial size to resize inputs/targets to (e.g., 64).",
    )
    parser.add_argument(
        "--train-inference-resize",
        type=int,
        default=None,
        help="Optional spatial size to downsample inputs before the model forward; loss is still computed at native resolution.",
    )
    parser.add_argument(
        "--train-crop-size",
        type=int,
        default=None,
        help="Enable random square crops of this size for training batches (patch training).",
    )
    parser.add_argument(
        "--val-crop-size",
        type=int,
        default=None,
        help="Slide a validation window of this square size at eval time (defaults to train crop size when unset).",
    )
    parser.add_argument(
        "--val-crop-overlap",
        type=float,
        default=None,
        help="Fractional overlap (0.0-<1.0) between sliding-window validation patches.",
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
        "--ram-cache",
        action="store_true",
        help="Keep decoded samples in RAM after first access (may require large memory).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the most recent epoch checkpoint for the run.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Python logging level (e.g., DEBUG, INFO, WARNING).",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        help="Model architecture variant to use (baseline, unet_lite).",
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
        choices=["group", "rms", "none"],
        help="Normalization layer for UNet-lite (group, rms, none).",
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
        "--lambda-ergas",
        type=float,
        default=None,
        help="Weight for ERGAS loss term (default 0.0).",
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
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=None,
        help="Learning rate scheduler to use (none, cosine).",
    )
    parser.add_argument(
        "--scheduler-warmup-epochs",
        type=int,
        default=None,
        help="Number of warmup epochs before cosine decay.",
    )
    parser.add_argument(
        "--scheduler-warmup-start-factor",
        type=float,
        default=None,
        help="Starting factor for warmup relative to base LR (default 0.1).",
    )
    parser.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=None,
        help="Minimum LR reached by the cosine scheduler.",
    )
    parser.add_argument(
        "--aug-rotate90",
        action="store_true",
        help="Enable random 90-degree rotations for train samples.",
    )
    parser.add_argument(
        "--aug-resized-crop",
        action="store_true",
        help="Enable mild random resized crop jitter for train samples.",
    )
    parser.add_argument(
        "--aug-intensity-jitter",
        action="store_true",
        help="Enable mild brightness/contrast jitter for train samples.",
    )
    parser.add_argument(
        "--use-residual-head",
        action="store_true",
        help="Enable coarse+residual refinement head in UNet-lite.",
    )
    parser.add_argument(
        "--use-spectral-conv",
        action="store_true",
        help="Enable 1D spectral convolution after the 61-band head.",
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
        help="Dropout probability in bottleneck/decoder residual blocks.",
    )
    parser.add_argument(
        "--stochastic-depth-p",
        type=float,
        default=None,
        help="Stochastic depth drop probability for bottleneck/decoder blocks.",
    )
    parser.add_argument(
        "--use-bottleneck-attention",
        action="store_true",
        help="Enable compact channel attention in the UNet-lite bottleneck.",
    )
    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=None,
        help="Kernel size for UNet-lite residual blocks (default 3).",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) used to initialize model weights (no optimizer resume).",
    )
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow checkpoint/model mismatches when loading (missing layers stay at init, optimizer state falls back to fresh).",
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)


