# Track 1 Pipeline Walkthrough

This quick reference shows the exact commands to move from raw data to trained
model and validation metrics using the minimal `src/` skeleton.

---

## 1. Place the data

Unzip the Kaggle Track 1 download so the folders look like:

```
data/
└── track1/
    ├── train/{mosaic,hsi_61}
    ├── test-public/{mosaic,hsi_61}
    └── test_original/mosaic        # optional, no GT provided
```

No other preprocessing is required—the dataset class reads `.npy` mosaics and
`.h5` cubes directly.

---

## 2. (Optional) Adjust defaults

Open `src/config.py` if you want to change global defaults (batch size,
learning rate, hidden channels, etc.). Command-line flags in `train.py` and
`evaluate.py` override these values without editing the file.

---

## 3. (Optional) Build a resized disk cache

Repeatedly resizing the 61-band cubes on the fly is the main I/O cost. You can
precompute 64×64 (or any size) mosaics + cubes once:

```bash
python -m src.utils.build_cache \
  --data-root data/track1 \
  --cache-dir data/cache/track1 \
  --split train \
  --size 64
```

Run the command again for `--split val` if you want the validation cache too.
If you skip this step the trainer will lazily create cache files on demand.

---

## 4. Train a model

```bash
python -m src.train \
  --data-root data/track1 \
  --run-name demo-run \
  --epochs 10 \
  --batch-size 4 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --cache-dir data/cache/track1
```

### Native-resolution UNet-Lite (recommended starting point)

The new UNet-Lite downsamples inside the network (pixel-unshuffle + stride-2
stages) so you can keep 1024×1024 mosaics and losses aligned with the full-resolution targets.

```bash
python -m src.train \
  --run-name unet-lite-native-v1 \
  --model-variant unet_lite \
  --unet-base-channels 48 \
  --latent-channels 16 \
  --encoder-depth 4 \
  --epochs 40 \
  --batch-size 2 \
  --learning-rate 3e-4 \
  --lr-scheduler cosine \
  --scheduler-warmup-epochs 5 \
  --scheduler-warmup-start-factor 0.2 \
  --lambda-l1 1.0 \
  --lambda-sam 0.2 \
  --lambda-sid 0.1 \
  --lambda-srgb-l1 0.2 \
  --lambda-srgb-ssim 0.05
```

- Leave `--resize-to` unset to keep the dataset at native resolution.
- Use `--train-inference-resize` only if you need to temporarily downsample inputs before the model forward (losses are still computed on the original size).
- Control spectral head capacity with `--latent-channels`; `--encoder-depth` sets how many stride-2 stages the UNet uses (default = 3).

- Outputs:
  - training/validation progress in the console
  - checkpoints under `src/models/simple_cnn/runs/demo-run/checkpoints/`
  - per-epoch metrics appended (and deduplicated on resume) in `src/models/simple_cnn/runs/demo-run/metrics/metrics.json`, including timing units plus placeholders for `val_resized` (training validation) / `val_native`

Swap in your own architecture by editing `src/model.py`.

---

## 5. Evaluate a checkpoint

```bash
# Native validation sweep (logs → val_native)
python -m src.evaluate \
  --data-root data/track1 \
  --run-name demo-run \
  --num-workers 4 \
  --prefetch-factor 2 \
  --cache-dir data/cache/track1 \
  --resolution native \
  --all-checkpoints \
  --upsample-metrics \
  --max-checkpoints 5

# Resized validation sweep (logs → eval_resized)
python -m src.evaluate \
  --data-root data/track1 \
  --run-name demo-run \
  --num-workers 4 \
  --prefetch-factor 2 \
  --cache-dir data/cache/track1 \
  --resolution resized \
  --resize-to 64 \
  --all-checkpoints \
  --upsample-metrics \
  --max-checkpoints 5

# Optional: limit concurrent checkpoints when GPU memory is constrained
python -m src.evaluate \
  --data-root data/track1 \
  --run-name demo-run \
  --resolution native \
  --all-checkpoints \
  --upsample-metrics \
  --max-checkpoints 9 \
  --max-parallel-checkpoints 4
```

If `--checkpoint` is omitted, the script automatically loads
`src/models/simple_cnn/runs/<run-name>/checkpoints/model_best.pt`.
Each sweep appends results to the persistent `metrics/metrics.json` (`val_native` for native runs, `eval_resized` for resized sweeps). Use `--upsample-metrics` to keep model inference at the training resolution while still comparing predictions against native-resolution ground truth, and `--max-checkpoints` to subsample long training runs evenly (first/last/best are always included). The evaluator streams per-batch load vs inference timing so you can gauge throughput; combine with `--max-parallel-checkpoints` to trade a few extra passes for lower GPU memory usage. The output still includes loss together with SAM, SID, ERGAS, PSNR/SSIM on sRGB, and ΔE00 on the public validation split.

---

## 6. Generate plots

```bash
python -m src.utils.metrics.plot_metrics --run-dir src/models/simple_cnn/runs/demo-run
```

The script now organises outputs under `metrics/accuracy/`, `metrics/speed/`, and `metrics/comparisons/` (train-resized vs native overlays + redesigned `metrics_overview.png`). Rerun it whenever you re-evaluate a checkpoint—the existing PNGs are overwritten.

---

## 7. Extend as needed

- Add new metrics in `src/metrics.py`
- Introduce richer augmentations inside `src/data.py`
- Track experiments by picking a new `--run-name` (stored under `src/models/simple_cnn/runs/<name>/`)
- For full-resolution inference, call
  `model.predict_full_resolution(mosaic_tensor)`—it downsamples to the training
  size internally and upsamples the prediction back to the original H×W.
- Use `--cache-dir none` or `--no-write-cache` if you want to disable caching.

That’s it—no extra build steps or compilation commands are required.


