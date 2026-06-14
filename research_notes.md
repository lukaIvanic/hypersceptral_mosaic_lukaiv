# Research Notes — 2026 ICASSP Hyper-Object Challenge (Track 1)

Last updated: 2025-11-10

## 0) One-paragraph brief
Reconstruct a 61-band hyperspectral reflectance cube (400–1000 nm, 10 nm step) from a single-plane Bayer-style mosaic (RGGB) acquired at commodity-camera resolution. Mosaics are RAW-like (linear, pre-ISP). Training provides paired mosaic/HSI; public/private/hidden tests evaluate with the SSC score (spectral, spatial, color components). Goal: high-fidelity spectral reconstruction from low-cost inputs.

---

## 1) Challenge logistics
- Timeline
  - Training/Public/Private test release: available now
  - Final code submission deadline: Nov 25, 2025
  - Winners announced: Nov 30, 2025
- Ranking metric
  - Overall: Spectral–Spatial–Color (SSC), ∈ [0,1] (higher is better)
  - Components:
    - Spectral: SAM, SID, ERGAS
    - Spatial: PSNR, SSIM (on standardized sRGB render; D65, CIE 1931 2°)
    - Color: ΔE00 (on standardized sRGB render)
  - Note: Exact SSC combination/weights are not stated in the paste; treat as a black box but optimize its components.

---

## 2) Data overview (Track 1)
- Scope
  - 61-band reflectance cubes (float32, [0,1]), wavelengths 400, 410, …, 1000 nm
  - Single-channel mosaic (Bayer RGGB), float32, [0,1], linear (pre-ISP)
  - Illumination: indoor lighting near D65; diverse everyday objects
- Size
  - Total ≈ 39 GB for Track 1
  - Train set size: 167 samples
- Splits and layout (as described)
  ```
  2026-Hyper-Object-Data/
  |-- train/
  |   |-- mosaic/        # {id}.npy   (H,W[,1])  float32, linear
  |   `-- hsi_61/        # {id}.h5    (H,W,61)   dataset "cube", float32 in [0,1]
  |-- test_public/
  |   |-- mosaic/        # {id}.npy
  |   `-- hsi_61/        # {id}.h5    (H,W,61)   (public GT for local eval)
  `-- test_private/
      `-- mosaic/        # {id}.npy
  ```

---

## 3) Mosaic generation pipeline (dataset’s stated method)
From the challenge page (used to produce the released mosaics):
1) Reflectance → linear RGB (D65)
   - Convert reflectance R(λ) to XYZ via CIE 1931 2° CMFs with D65; unit-normalized so a perfect diffuser yields Y=1.
   - Convert XYZ → linear sRGB (D65) via the standard 3×3 matrix.
   - Stay linear; no gamma/white-balance/tone mapping; clip negatives to 0.
2) Bayer mosaicing (RGGB)
   - Apply 2×2 RGGB CFA: each pixel samples one of {R,G,B} by tile position.
   - Output is single-channel mosaic (RAW-like).
3) Normalization
   - Store as float32 in [0,1]. No demosaic, no color correction, no gamma, no sharpening, no JPEG.

Motivation: approximates a camera-like opto-electronic front-end while avoiding vendor ISP; keeps the learning task close to real mosaics (you must learn demosaicing + spectral reconstruction jointly or staged).

---

## 4) HSI ground truth specifics
- Bands: 61 (400–1000 nm, 10 nm step)
- Origin: derived by subsampling/integration from an original 448-band HSI (not simple slicing; preserves band energy more faithfully)
- Calibration: radiometric calibration applied (white/dark references, central-ROI medians), robust to minor vignetting; HSI is reflectance in [0,1]

---

## 5) File formats and keys
- Mosaic: `.npy` (NumPy array), float32 in [0,1], shape (H,W) or (H,W,1)
- HSI: `.h5` (HDF5), dataset `'cube'` with shape (H,W,61), float32 in [0,1]
- Optional: dataset `'wavelengths'` with shape (61,)

Why this split:
- `.npy` is fast/simple for a single array (mosaic).
- `.h5` scales well for multi-band cubes (+ optional metadata), supports chunking/compression.

---

## 6) Submission specification
- Upload a .zip containing:
  - `submission.csv` at zip root, columns:
    - `id` → test sample ID without extension (e.g., `Category-1_a_0007`)
    - `prediction` → 0 (placeholder required by Kaggle)
  - For each test ID: `<id>.npz` with key `cube` of shape (H, W, 61), float32 in [0,1]
- Strictness:
  - Filenames must match private-test IDs exactly
  - `submission.csv` must be at root of zip
  - Each `.npz` must have key `cube`; missing/wrong shapes fail

Example save:
```python
np.savez("Category-1_a_0007.npz", cube=cube_hwc)  # (H,W,61), float32
```

---

## 7) Evaluation, what matters practically
- Optimize SSC by improving:
  - Spectral fidelity (SAM↓, SID↓, ERGAS↓)
  - Spatial fidelity (PSNR↑, SSIM↑ on standardized sRGB rendering)
  - Color accuracy (ΔE00↓ on standardized sRGB render)
- Important implications
  - Output must represent reflectance (not radiance), within [0,1].
  - Rendering sRGB for PSNR/SSIM/ΔE00 uses D65 + CIE 1931 2° pipeline; be consistent when computing local metrics for validation.
  - Spatial sharpness without spectral distortion is key (avoid over-smoothing spectra to gain PSNR).

---

## 8) Baseline visualization intuition (for green apple)
- Reflectance peaks near ~550 nm (green): band ~15 is bright
- Lower reflectance near ~650–700 nm (red edge): band ~30 dimmer than green
- NIR (>700 nm) rises again: band ~45 (≈850 nm) bright, but RGB mosaic cannot “see” it (RGB sensitivities drop in NIR)
- Pseudo-RGB from HSI: a practical mapping that often looks natural is (R,G,B) = (30,15,0)

---

## 9) Practical workflow with our skeleton
- Optional cache build
  `python -m tools.build_cache --data-root data/track1 --cache-dir cache/track1 --split train --size 64`
- Visual checks
  - Mosaic only (color demosaic preview):
    `python -m src.utils.preview --data-root data/track1 --index 0 --input-only`
  - HSI pseudo-RGB (30,15,0):
    `python -m src.utils.preview --data-root data/track1 --index 0 --bands 30 15 0`
  - Single band (e.g., 30):
    `python -m src.utils.preview --data-root data/track1 --index 0 --band 30`
- Train
  - `python -m src.train --data-root data/track1 --run-name exp001 --epochs 50 --batch-size 4 --num-workers 4 --prefetch-factor 2 --cache-dir data/cache/track1`
  - Use `--cache-dir none` or `--no-write-cache` to disable disk caching.
- Evaluate (public val)
  - `python -m src.evaluate --data-root data/track1 --run-name exp001 --num-workers 4 --prefetch-factor 2 --cache-dir data/cache/track1`
  - `--checkpoint` is optional; when omitted the script uses `…/runs/<run-name>/checkpoints/model_best.pt`.
- Submission packaging (ensure reflectance in [0,1], shapes (H,W,61)): follow Section 6
- Full-res inference
  - `pred = model.predict_full_resolution(mosaic_tensor)` downsamples to the training resolution internally and upsamples back to the original size.

---

## 10) Modeling ideas (research directions)
- Architectures
  - Plain CNN with residual blocks; UNet with skip connections; multi-scale receptive fields
  - Pixel-unshuffle/unpack to exploit CFA structure (RGGB → 4 maps) then conv, then fuse
  - Spectral decoders: predict PCA coefficients (basis learned or fixed) then reconstruct 61 bands
  - 3D convs (C×H×W) or 2D convs with spectral attention (lightweight)
- Training losses
  - L1 + SAM (common strong baseline)
  - Add spectral smoothness/TV along bands
  - Optional color-consistency term: render sRGB (D65) from predicted cube and compare to demosaiced RGB proxy (careful: proxy is synthetic)
- Data handling
  - Random crops/patches; flips; patch-size balance for VRAM
  - Normalization: inputs/targets are already in [0,1], keep consistent
- Validation strategy
  - Local metrics: SAM, SID, ERGAS, PSNR, SSIM; render sRGB with D65
  - Track correlation with SSC on public/private set (if available)
- Ablations
  - With/without pixel-unshuffle preprocessing
  - PCA targets vs direct 61-band regression
  - Loss weight for SAM; add spectral TV

---

## 11) Risks, caveats, open questions
- SSC exact aggregation weights may be hidden; optimize components, monitor generalization
- Illumination approximated as D65 in mosaic generation; real-world mismatch could affect deployment
- Mosaics are simulated, not captured with a vendor camera; good for alignment, but spectral responses are “sRGB-like,” not specific to any manufacturer
- Varying image sizes: write code without hardcoding shapes

---

## 12) Leaderboard snapshot (from pasted page)
- Top: ~0.64–0.61 SSC; mid ~0.41; lower ~0.13 with very small/brief training
- Takeaway: 0.6+ likely requires careful modeling/training; naive short CPU training yields weak scores (~0.13)

---

## 13) Quick checklist
- [ ] Verify data layout matches Section 2
- [ ] Sanity-visualize mosaics and bands (Section 9)
- [ ] Start with L1+SAM, simple CNN; ensure output clamp [0,1]
- [ ] Track spectral smoothness; avoid artifacts that hurt ΔE00 after rendering
- [ ] Package a submission (Section 6) and smoke test zip structure
