
# Track 1 Minimal Skeleton

This repository is a **minimal training and evaluation skeleton** for the 2026 ICASSP Hyper-Object Challenge — Track 1 (mosaic → 61-band hyperspectral cube). All legacy code and multi-track abstractions were removed so you can focus on a single, easy-to-follow pipeline.

- Challenge site: https://hyper-object.github.io/
- Kaggle competition: https://www.kaggle.com/competitions/2026-icassp-hyper-object-challenge-track-1

---

## Why this structure?

The project is intentionally organised around the smallest set of moving parts required to go from raw data to a trained model and back to validation metrics:

- `src/config.py` – One place to store hyperparameters and paths so train/eval stay in sync.
- `src/data.py` – Track‑1 specific dataset + dataloader factory with simple, coordinated augmentation.
- `src/model.py` – A single CNN baseline you can swap out for your own architecture.
- `src/train.py` – End-to-end training loop (argparse, logging, checkpoints).
- `src/evaluate.py` – Loads a checkpoint and reports validation metrics.
- `src/metrics.py` – Small bundle of reconstruction metrics (MAE, MSE, PSNR, SAM).

Everything else lives under `data/track1/`, matching the Kaggle download layout. The goals are clarity and hackability: each file is self-contained, short, and documented so you can modify or replace it without digging through a deep module hierarchy.

---

## Repository layout

```
.
├── data/track1/                # Place the Kaggle Track 1 data here
│   ├── train/{mosaic,hsi_61}
│   ├── test-public/{mosaic,hsi_61}
│   └── test_original/mosaic    # Optional, no ground truth provided
└── src/
    ├── __init__.py
    ├── PIPELINE.md             # Step-by-step run instructions
    ├── config.py               # TrainConfig dataclass
    ├── data.py                 # Dataset + dataloaders
    ├── evaluate.py             # Validation script
    ├── metrics.py              # Metric helpers
    ├── model.py                # SimpleHSIModel baseline
    └── train.py                # Training CLI
```

Feel free to add your own modules under `src/` as the project grows (e.g., `src/augment.py`, `src/losses.py`).

---

## Environment

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118  # adjust for your CUDA
pip install numpy h5py colour-science matplotlib
```

Only standard scientific Python libraries are used. The starter model trains on CPU, but a GPU is recommended for meaningful experiments.

---

## Prepare the data

1. Download the Track 1 dataset from Kaggle.
2. Unzip so the mosaics and HSI cubes live under `data/track1/` as shown above.

The validation script expects the public validation cubes (`test-public/hsi_61`). Make sure they are present before running evaluation.

---

## Train

```bash
python -m src.train \
    --data-root data/track1 \
    --run-name experiment-001 \
    --epochs 20 \
    --batch-size 4 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --cache-dir data/cache/track1
```

- Checkpoints are written to `src/models/simple_cnn/runs/experiment-001/checkpoints/`.
- Epoch records (train timings, losses, and validation summaries) accumulate in `src/models/simple_cnn/runs/experiment-001/metrics/metrics.json`, now annotated with explicit units plus `val_resized` (training-time validation) and `val_native` (post-train evaluation) sections.
- After training, run the evaluator (see below) for each resolution you care about so the comparison plots have data.
- Refresh visualisations with `python -m src.utils.metrics.plot_metrics --run-dir src/models/simple_cnn/runs/experiment-001`; the script now produces structured folders: `metrics/accuracy/`, `metrics/speed/`, and `metrics/comparisons/` (including an overhauled `metrics_overview.png`).
- Adjust hyperparameters via CLI flags or by editing `TrainConfig`.
- Replace `SimpleHSIModel` inside `src/model.py` with your own network to iterate quickly.

> Tip: run `python -m src.utils.build_cache --data-root data/track1 --cache-dir data/cache/track1 --split train --size 64`
> (and `--split val`) once to precompute the resized cache; subsequent training runs will load the 64×64 mosaics/HSI cubes directly.

---

## Evaluate a checkpoint

```bash
# Native-resolution validation across every checkpoint (logs → val_native)
python -m src.evaluate \
    --data-root data/track1 \
    --run-name experiment-001 \
    --resolution native \
    --all-checkpoints \
    --upsample-metrics \
    --max-checkpoints 5

# 64×64 validation to mirror the training resize (logs → eval_resized)
python -m src.evaluate \
    --data-root data/track1 \
    --run-name experiment-001 \
    --resolution resized \
    --resize-to 64 \
    --all-checkpoints \
    --upsample-metrics \
    --max-checkpoints 5

# (Optional) limit concurrent models to reduce GPU memory pressure
python -m src.evaluate \
    --data-root data/track1 \
    --run-name experiment-001 \
    --resolution native \
    --all-checkpoints \
    --upsample-metrics \
    --max-checkpoints 9 \
    --max-parallel-checkpoints 4
```

- Evaluation now streams per-batch timing (disk load vs inference) while running every requested checkpoint on each batch before touching the next image. Use `--max-parallel-checkpoints` when GPU memory is tight; smaller slices reuse the same disk reads at the cost of repeated passes.
- Add `--upsample-metrics` to downsample inputs to the training resolution for inference while still scoring against native-resolution ground truth (model predictions are upsampled before metric calculation).
- Combine `--all-checkpoints` with `--max-checkpoints` to subsample long training runs—first, last, and the checkpoint tagged `model_best.pt` are always included, with the rest spaced evenly across the schedule.
- Results are appended to `metrics/metrics.json`, so you can compare `val_native` (post-train native validation) vs `val_resized` (in-training validation). Optional resized evaluation sweeps land in `eval_resized`. Use `--epoch` to override the history epoch for single-checkpoint runs or `--metrics-file` to log into a custom path.
- The evaluator reports the full SSC-aligned set (`SAM`, `SID`, `ERGAS`, `PSNR_SRGB`, `SSIM_SRGB`, `DeltaE00`) plus any extra metrics you request via `--metrics`.
- Target validation ranges for the key SSC components remain summarised in `METRIC_TARGETS.md`; the plot script now overlays these thresholds for quick visual triage.

---

## Need a quick checklist?

See `src/PIPELINE.md` for a concise, step-by-step walkthrough (data placement → train → evaluate) with the exact commands used in this skeleton.

---

## Extending the skeleton

- **New models** – Drop in a class under `src/model.py` or a new module and instantiate it in `train.py`.
- **Custom losses / metrics** – Add helpers under `src/metrics.py` or create a new `src/losses.py`, then wire them into the training loop.
- **Augmentations** – Expand the augmentation hook inside `src/data.py` or plug in a different transform callable.

The guiding principle is to keep responsibilities explicit and modules short. Build outward only when the current file becomes hard to reason about.

---

Happy hacking! Submit issues or suggestions if the skeleton can be simplified further.***
