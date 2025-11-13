from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import TrainConfig
from src.data import Track1Dataset
from src.evaluate import load_model, run_model_with_resize


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference on validation samples and generate per-pixel MAE heatmaps. "
            "Requires ground-truth availability (validation/public test split)."
        )
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Training run name to locate checkpoints under src/models/simple_cnn/runs/<run-name>/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a specific checkpoint (.pt). Overrides --run-name when provided.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Load checkpoints with strict=False to allow missing/unexpected keys.",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root override (defaults to TrainConfig.data_root).")
    parser.add_argument("--device", type=str, default=None, help="Torch device to run inference on (e.g., cuda, cuda:0).")
    parser.add_argument("--model-variant", type=str, default=None, help="Model variant override (baseline, unet_lite, ...).")
    parser.add_argument("--hidden-channels", type=int, default=None, help="Hidden channel width override for baseline variant.")
    parser.add_argument("--unet-base-channels", type=int, default=None, help="Base channels override for UNet variants.")
    parser.add_argument("--latent-channels", type=int, default=None, help="Latent channel width for UNet-lite variants.")
    parser.add_argument("--encoder-depth", type=int, default=None, help="Number of stride-2 encoder stages for UNet-lite.")
    parser.add_argument("--coarse-channels", type=int, default=None, help="Number of coarse spectral channels before interpolation.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for validation loader (default: 1).")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Prefetch factor used when num-workers > 0 (default: 2).",
    )
    parser.add_argument(
        "--inference-resize",
        type=int,
        default=0,
        help="Optional spatial size to downsample inputs during inference (0 keeps native resolution).",
    )
    parser.add_argument(
        "--resize-to",
        type=int,
        default=0,
        help="Optional spatial size to downsample BOTH mosaics and ground truth targets before inference (0 keeps native resolution).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Limit number of samples processed (0 means all, applied after --sample-ids filtering).",
    )
    parser.add_argument(
        "--sample-ids",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of sample ids to process (e.g., Category-1_a_0008 Category-1_a_0022).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where heatmap PNGs (and optional raw arrays) will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save the raw MAE heatmap as <id>_pixel_mae.npy alongside the PNG.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap name for heatmaps (default: coolwarm).",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Fix the upper bound for heatmap color scaling. If omitted, derived from --percentile.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Percentile used to set vmax when --vmax is not provided (default: 99.5).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved heatmap figures (default: 150).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(6.0, 5.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Matplotlib figure size in inches (default: 6 5).",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Disable rendering a colorbar alongside the heatmap.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Dataset cache directory override (use 'none' to disable).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading from the dataset cache.",
    )
    parser.add_argument(
        "--no-write-cache",
        action="store_true",
        help="Avoid writing resized samples to cache.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="How often to print progress information in batches (<=0 disables).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_checkpoint(
    args: argparse.Namespace,
    model_runs_root: Path,
) -> Path:
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


def _subset_dataset(
    dataset: Track1Dataset,
    sample_ids: Optional[Sequence[str]],
    sample_limit: int,
) -> Tuple[torch.utils.data.Dataset, List[str]]:
    if sample_ids:
        available = {sid: idx for idx, sid in enumerate(dataset.ids)}
        missing = sorted({sid for sid in sample_ids if sid not in available})
        if missing:
            raise ValueError(f"Requested sample ids not found in validation split: {missing}")
        ordered_indices = [available[sid] for sid in sample_ids]
        selected_ids = list(sample_ids)
    else:
        ordered_indices = list(range(len(dataset)))
        selected_ids = [dataset.ids[idx] for idx in ordered_indices]

    if sample_limit > 0:
        ordered_indices = ordered_indices[:sample_limit]
        selected_ids = selected_ids[:sample_limit]

    if len(ordered_indices) == len(dataset):
        return dataset, selected_ids
    return Subset(dataset, ordered_indices), selected_ids


def _format_size(tensor: torch.Tensor) -> str:
    return "x".join(str(dim) for dim in tensor.shape)


@torch.no_grad()
def generate_heatmaps(args: argparse.Namespace) -> None:
    cfg = TrainConfig()
    cfg.resize_to = None  # default to native resolution
    if args.data_root is not None:
        cfg.data_root = Path(args.data_root).expanduser().resolve()
    if args.device is not None:
        cfg.device = args.device
    if args.hidden_channels is not None:
        cfg.hidden_channels = args.hidden_channels
    if args.model_variant is not None:
        cfg.model_variant = args.model_variant.lower()
    if args.unet_base_channels is not None:
        cfg.unet_base_channels = args.unet_base_channels
    if args.latent_channels is not None:
        cfg.latent_channels = args.latent_channels
    if args.encoder_depth is not None:
        cfg.encoder_depth = max(1, args.encoder_depth)
    if args.coarse_channels is not None:
        cfg.coarse_output_channels = max(1, args.coarse_channels)
    if args.cache_dir is not None:
        cfg.cache_dir = None if args.cache_dir.lower() in {"", "none"} else Path(args.cache_dir)
    if args.no_cache:
        cfg.cache_dir = None
    if args.no_write_cache:
        cfg.write_cache = False

    inference_resize: Optional[int] = args.inference_resize
    if inference_resize is not None and inference_resize <= 0:
        inference_resize = None

    device = torch.device(cfg.device)
    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_runs_root = Path(__file__).resolve().parent.parent / "src" / "models" / "simple_cnn" / "runs"
    checkpoint_path = _resolve_checkpoint(args, model_runs_root)
    strict = not args.allow_partial
    model, info = load_model(checkpoint_path, cfg, device, strict=strict)
    model.eval()
    print(
        f"[Heatmap] Loaded checkpoint='{checkpoint_path.name}' "
        f"(variant={info.get('loaded_variant', cfg.model_variant)}) "
        f"| device={device}"
    )

    dataset_resize: Optional[int] = None
    resize_to_arg = getattr(args, "resize_to", 0)
    if resize_to_arg is not None and resize_to_arg > 0:
        dataset_resize = int(resize_to_arg)

    dataset = Track1Dataset(
        root=cfg.data_root,
        split="val",
        resize_to=dataset_resize,
        cache_dir=cfg.cache_dir,
        write_cache=cfg.write_cache,
    )
    if not getattr(dataset, "targets_available", False):
        raise RuntimeError("Validation split ground truth is unavailable. Heatmap inspection requires GT cubes.")

    subset_dataset, selected_ids = _subset_dataset(dataset, args.sample_ids, args.sample_limit)

    loader_kwargs = {
        "batch_size": max(1, args.batch_size),
        "shuffle": False,
        "num_workers": max(0, args.num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = max(1, args.prefetch_factor)

    loader = DataLoader(subset_dataset, **loader_kwargs)
    if len(loader) == 0:
        print("[Heatmap] No samples selected; exiting.")
        return

    resize_label = dataset_resize if dataset_resize is not None else "native"
    print(
        f"[Heatmap] Selected {len(selected_ids)} sample(s) | dataset_resize={resize_label} | "
        f"batch_size={loader_kwargs['batch_size']}"
    )

    percentile = args.percentile
    if percentile is not None and not (0.0 <= percentile <= 100.0):
        raise ValueError("--percentile must be within [0, 100].")

    total_pixel_sum = 0.0
    total_pixel_count = 0
    global_max = 0.0
    per_sample_means: List[Tuple[str, float]] = []
    band_sums: Optional[torch.Tensor] = None
    band_pixel_count: int = 0

    processed = 0
    cmap = plt.get_cmap(args.cmap)
    for batch_idx, batch in enumerate(loader, start=1):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        preds = run_model_with_resize(
            model,
            inputs,
            inference_resize,
            final_shape=targets.shape[-2:],
        )
        preds = preds.clamp(0.0, 1.0)

        diff = torch.abs(preds - targets)
        mae_maps = diff.mean(dim=1)  # (B, H, W)

        if band_sums is None:
            band_sums = diff.sum(dim=(0, 2, 3)).cpu()
        else:
            band_sums += diff.sum(dim=(0, 2, 3)).cpu()
        band_pixel_count += diff.shape[0] * diff.shape[2] * diff.shape[3]

        batch_ids: Iterable[str]
        ids_field = batch["id"]
        if isinstance(ids_field, list):
            batch_ids = ids_field
        elif isinstance(ids_field, (tuple, torch.Tensor)):
            if isinstance(ids_field, torch.Tensor):
                batch_ids = [dataset.ids[int(idx)] for idx in ids_field.tolist()]
            else:
                batch_ids = list(ids_field)
        else:
            batch_ids = [ids_field]  # type: ignore[list-item]

        for offset, sample_id in enumerate(batch_ids):
            mae_map = mae_maps[offset].detach().cpu().numpy()
            sample_mean = float(mae_map.mean())
            sample_max = float(mae_map.max())

            total_pixel_sum += float(mae_map.sum())
            total_pixel_count += int(mae_map.size)
            global_max = max(global_max, sample_max)
            per_sample_means.append((sample_id, sample_mean))
            processed += 1

            if args.vmax is not None:
                vmax = args.vmax
            else:
                if mae_map.size == 0:
                    vmax = 0.0
                else:
                    vmax = float(np.percentile(mae_map, percentile)) if percentile else float(mae_map.max())
            if vmax <= 0:
                vmax = float(mae_map.max()) if mae_map.size else 1e-8
            if vmax <= 0:
                vmax = 1e-6

            fig, ax = plt.subplots(figsize=tuple(args.figsize))
            im = ax.imshow(mae_map, cmap=cmap, vmin=0.0, vmax=vmax)
            ax.set_title(f"{sample_id} | mean={sample_mean:.5f} | max={sample_max:.5f}")
            ax.set_axis_off()
            if not args.no_colorbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Per-pixel MAE")
            fig.tight_layout()

            png_path = output_dir / f"{sample_id}_pixel_mae.png"
            if png_path.exists() and not args.overwrite:
                print(f"[Heatmap] Skipping existing file (use --overwrite): {png_path.name}")
            else:
                fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
                print(f"[Heatmap] Saved heatmap -> {png_path}")
            plt.close(fig)

            band_dir = output_dir / sample_id / "bands"
            band_dir.mkdir(parents=True, exist_ok=True)
            band_maps = diff[offset].detach().cpu().numpy()
            for band_idx in range(band_maps.shape[0]):
                band_map = band_maps[band_idx]
                if args.vmax is not None:
                    band_vmax = args.vmax
                else:
                    if band_map.size == 0:
                        band_vmax = 0.0
                    else:
                        band_vmax = (
                            float(np.percentile(band_map, percentile))
                            if percentile
                            else float(band_map.max())
                        )
                if band_vmax <= 0:
                    band_vmax = float(band_map.max()) if band_map.size else 1e-8
                if band_vmax <= 0:
                    band_vmax = 1e-6

                fig_band, ax_band = plt.subplots(figsize=tuple(args.figsize))
                im_band = ax_band.imshow(band_map, cmap=cmap, vmin=0.0, vmax=band_vmax)
                ax_band.set_title(
                    f"{sample_id} | band={band_idx + 1:02d} | mean={band_map.mean():.5f}"
                )
                ax_band.set_axis_off()
                if not args.no_colorbar:
                    fig_band.colorbar(
                        im_band, ax=ax_band, fraction=0.046, pad=0.04, label="MAE"
                    )
                fig_band.tight_layout()

                band_png = band_dir / f"band_{band_idx + 1:02d}_mae.png"
                if band_png.exists() and not args.overwrite:
                    print(f"[Heatmap] Skipping existing band file (use --overwrite): {band_png}")
                else:
                    fig_band.savefig(band_png, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig_band)

                if args.save_npy:
                    band_npy = band_dir / f"band_{band_idx + 1:02d}_mae.npy"
                    if band_npy.exists() and not args.overwrite:
                        print(
                            f"[Heatmap] Skipping existing band .npy (use --overwrite): {band_npy}"
                        )
                    else:
                        np.save(band_npy, band_map.astype(np.float32))

            if args.save_npy:
                npy_path = output_dir / f"{sample_id}_pixel_mae.npy"
                if npy_path.exists() and not args.overwrite:
                    print(f"[Heatmap] Skipping existing .npy (use --overwrite): {npy_path.name}")
                else:
                    np.save(npy_path, mae_map.astype(np.float32))

        if args.progress_every > 0 and (
            batch_idx % args.progress_every == 0 or batch_idx == len(loader)
        ):
            print(
                f"[Heatmap] Progress: batch {batch_idx}/{len(loader)} | processed {processed} samples | "
                f"inputs { _format_size(inputs) } -> preds { _format_size(preds) }"
            )

    if processed == 0:
        print("[Heatmap] No samples processed.")
        return

    dataset_info = (
        f"ids={len(selected_ids)} (limit applied)" if processed < len(selected_ids) else f"ids={processed}"
    )
    overall_mean = total_pixel_sum / max(total_pixel_count, 1)
    print(
        f"[Heatmap] Summary :: samples={processed} ({dataset_info}) | "
        f"overall_mean_mae={overall_mean:.6f} | global_max={global_max:.6f}"
    )
    per_sample_means.sort(key=lambda item: item[1], reverse=True)
    top_preview = per_sample_means[: min(5, len(per_sample_means))]
    for sample_id, mean_value in top_preview:
        print(f"[Heatmap] Top mean pixels :: {sample_id} -> {mean_value:.6f}")

    if band_sums is not None and band_pixel_count > 0:
        band_avg = (band_sums / band_pixel_count).numpy()
        band_indices = np.arange(1, band_avg.shape[0] + 1, dtype=np.int32)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(band_indices, band_avg, marker="o")
        ax.set_xlabel("Spectral Band")
        ax.set_ylabel("Average MAE (per pixel)")
        ax.set_title("Average per-band MAE across processed samples")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        band_plot_path = output_dir / "band_mae_summary.png"
        fig.savefig(band_plot_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Heatmap] Saved per-band summary plot -> {band_plot_path}")
        np.save(output_dir / "band_mae_summary.npy", band_avg.astype(np.float32))


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    generate_heatmaps(args)


if __name__ == "__main__":
    main()


