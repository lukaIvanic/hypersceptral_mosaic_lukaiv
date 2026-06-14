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

Outputs:
- training/validation progress in the console
- checkpoints under `src/models/simple_cnn/runs/demo-run/checkpoints/`

Swap in your own architecture by editing `src/model.py`.

---

## 5. Evaluate a checkpoint

```bash
python -m src.evaluate \
  --data-root data/track1 \
  --run-name demo-run \
  --num-workers 4 \
  --prefetch-factor 2 \
  --cache-dir data/cache/track1
```

If `--checkpoint` is omitted, the script automatically loads
`src/models/simple_cnn/runs/<run-name>/checkpoints/model_best.pt`.
It reports loss, MAE, MSE, PSNR, and SAM on the public validation split
(`test-public`).

---

## 6. Extend as needed

- Add new metrics in `src/metrics.py`
- Introduce richer augmentations inside `src/data.py`
- Track experiments by picking a new `--run-name` (stored under `src/models/simple_cnn/runs/<name>/`)
- For full-resolution inference, call
  `model.predict_full_resolution(mosaic_tensor)`—it downsamples to the training
  size internally and upsamples the prediction back to the original H×W.
- Use `--cache-dir none` or `--no-write-cache` if you want to disable caching.

That’s it—no extra build steps or compilation commands are required.
