from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import Track1Dataset
from .evaluate import load_model
from .utils.inference import run_model_with_resize


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export hyperspectral predictions for the competition test split. "
            "Generates one <id>.npz (key='cube') per sample and an optional "
            "submission.csv stub."
        )
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root (defaults to TrainConfig.data_root).")
    parser.add_argument("--run-name", type=str, default=None, help="Training run name whose checkpoints should be used.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit checkpoint path to load instead of run-name default.")
    parser.add_argument("--model-variant", type=str, default=None, help="Model variant override (e.g. unet_lite).")
    parser.add_argument("--hidden-channels", type=int, default=None, help="Hidden channel width for baseline variant.")
    parser.add_argument(
        "--unet-base-channels",
        type=int,
        default=None,
        help="Base channel count for UNet variants (mirrors train/evaluate flag).",
    )
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=None,
        help="Latent channel width for UNet-lite spectral head.",
    )
    parser.add_argument(
        "--encoder-depth",
        type=int,
        default=None,
        help="Number of stride-2 encoder stages for UNet-lite.",
    )
    parser.add_argument(
        "--coarse-channels",
        type=int,
        default=None,
        help="Number of coarse spectral channels before interpolation.",
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
        help="Kernel size for spectral 1D convolution.",
    )
    parser.add_argument(
        "--decoder-dropout",
        type=float,
        default=None,
        help="Dropout probability for decoder/bottleneck residual blocks.",
    )
    parser.add_argument(
        "--stochastic-depth-p",
        type=float,
        default=None,
        help="Stochastic depth probability for decoder/bottleneck blocks.",
    )
    parser.add_argument(
        "--use-bottleneck-attention",
        action="store_true",
        help="Enable bottleneck attention module (must match training).",
    )
    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=None,
        help="Kernel size for UNet-lite residual blocks (default 3; must match training).",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device to run inference on.")
    parser.add_argument(
        "--inference-resize",
        type=int,
        default=None,
        help="Optional spatial size to downsample inputs to before model inference.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export loader (defaults to 1).")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch factor when num-workers > 0.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory override (use 'none' to disable). Test split caching is optional.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset cache usage.")
    parser.add_argument("--no-write-cache", action="store_true", help="Avoid writing any new cache files.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split name to export (default: 'test'). Must exist under data root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where <id>.npz predictions (and optional submission.csv) are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <id>.npz files instead of skipping them.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Load checkpoints with strict=False, allowing missing/unexpected keys.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="How often (in batches) to log progress information (<=0 disables).",
    )
    parser.add_argument(
        "--write-submission-csv",
        action="store_true",
        help="Also emit submission.csv with placeholder predictions (required by Kaggle uploads).",
    )
    parser.add_argument(
        "--swap-hw",
        action="store_true",
        help="Transpose predictions before saving so output shape becomes (W, H, C) instead of (H, W, C).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_checkpoint(args: argparse.Namespace, model_runs_root: Path) -> Path:
    if args.checkpoint is not None:
        ckpt_path = args.checkpoint.expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    if args.run_name is None:
        raise ValueError("Either --run-name or --checkpoint must be provided.")

    run_dir = model_runs_root / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    best_path = ckpt_dir / "model_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_path}. Pass --checkpoint explicitly.")
    return best_path


def export_predictions(args: argparse.Namespace) -> None:
    cfg = TrainConfig()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.model_variant is not None:
        cfg.model_variant = args.model_variant.lower()
    if args.hidden_channels is not None:
        cfg.hidden_channels = args.hidden_channels
    if args.unet_base_channels is not None:
        cfg.unet_base_channels = args.unet_base_channels
    if args.latent_channels is not None:
        cfg.latent_channels = args.latent_channels
    if args.encoder_depth is not None:
        cfg.encoder_depth = max(1, args.encoder_depth)
    if args.coarse_channels is not None:
        cfg.coarse_output_channels = max(1, args.coarse_channels)
    if args.use_residual_head:
        cfg.use_residual_head = True
    if args.use_spectral_conv:
        cfg.use_spectral_conv = True
    if args.spectral_conv_kernel_size is not None:
        cfg.spectral_conv_kernel_size = max(1, int(args.spectral_conv_kernel_size))
    if args.decoder_dropout is not None:
        cfg.decoder_dropout = max(0.0, float(args.decoder_dropout))
    if args.stochastic_depth_p is not None:
        cfg.stochastic_depth_p = max(0.0, float(args.stochastic_depth_p))
    if args.use_bottleneck_attention:
        cfg.use_bottleneck_attention = True
    if args.conv_kernel_size is not None:
        cfg.conv_kernel_size = max(1, int(args.conv_kernel_size))
    if args.device is not None:
        cfg.device = args.device
    if args.cache_dir is not None:
        cfg.cache_dir = None if args.cache_dir.lower() in {"", "none"} else Path(args.cache_dir)
    if args.no_cache:
        cfg.cache_dir = None
    if args.no_write_cache:
        cfg.write_cache = False

    inference_resize: Optional[int] = args.inference_resize
    if inference_resize is not None and inference_resize <= 0:
        inference_resize = None
    cfg.resize_to = inference_resize

    device = torch.device(cfg.device)
    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_runs_root = Path(__file__).resolve().parent / "models" / "simple_cnn" / "runs"
    checkpoint_path = _resolve_checkpoint(args, model_runs_root)

    strict = not args.allow_partial
    model, ckpt_info = load_model(checkpoint_path, cfg, device, strict=strict)
    model.eval()

    dataset = Track1Dataset(
        root=cfg.data_root,
        split=args.split,
        resize_to=None,  # always load native resolution for submission export
        cache_dir=cfg.cache_dir,
        write_cache=cfg.write_cache,
    )

    loader_kwargs = {
        "batch_size": max(1, args.batch_size),
        "shuffle": False,
        "num_workers": max(0, args.num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = max(2, args.prefetch_factor)

    loader = DataLoader(dataset, **loader_kwargs)
    total_batches = len(loader)
    total_images = len(dataset)
    print(
        f"[Export] Loaded checkpoint='{checkpoint_path.name}' "
        f"(variant={ckpt_info.get('loaded_variant', cfg.model_variant)}) "
        f"for split='{args.split}' | samples={total_images} | device={device}"
    )

    processed = 0
    skipped = 0
    for batch_idx, batch in enumerate(loader, start=1):
        mosaics: torch.Tensor = batch["input"].to(device, non_blocking=True)
        ids: List[str] = batch["id"]
        if not isinstance(ids, list):
            ids = [ids]

        final_shape = tuple(int(dim) for dim in mosaics.shape[-2:])
        preds = run_model_with_resize(model, mosaics, inference_resize, final_shape)
        preds = preds.detach().clamp_(0.0, 1.0).cpu()

        for sample_idx, sample_id in enumerate(ids):
            out_path = output_dir / f"{sample_id}.npz"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            cube_chw = preds[sample_idx]
            if args.swap_hw:
                cube_hwc = cube_chw.permute(2, 1, 0).contiguous().numpy().astype(np.float32)
            else:
                cube_hwc = cube_chw.permute(1, 2, 0).contiguous().numpy().astype(np.float32)
            np.savez(out_path, cube=cube_hwc)
            processed += 1

        if args.progress_every > 0 and (batch_idx % args.progress_every == 0 or batch_idx == total_batches):
            print(f"[Export] Batch {batch_idx}/{total_batches} | written={processed} | skipped={skipped}")

    print(f"[Export] Done. wrote={processed} | skipped={skipped} | output_dir={output_dir}")

    if args.write_submission_csv:
        csv_path = output_dir / "submission.csv"
        if csv_path.exists() and not args.overwrite:
            print(f"[Export] submission.csv exists — skipping (use --overwrite to replace).")
        else:
            all_ids = sorted(p.stem for p in output_dir.glob("*.npz") if p.name != "submission.csv")
            lines = ["id,prediction"]
            lines.extend(f"{sample_id},0" for sample_id in all_ids)
            csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"[Export] Wrote placeholder submission.csv with {len(all_ids)} rows.")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    export_predictions(args)


if __name__ == "__main__":
    main(sys.argv[1:])

