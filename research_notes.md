# Research Notes — 2026 ICASSP Hyper-Object Challenge (Track 1)

Last updated: 2025-11-18

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


---

## 14) Iterative improvement backlog (updated 2025-11-10)

### Workflow commitments
- Maintain all active ideas, experiments, and outcomes in this document.
- Execute backlog items sequentially; finish one change, evaluate it, and log metrics before moving to the next.
- When an experiment is required, ask the training agent (user) to run the specified command and report validation metrics so they can be recorded here.
- Keep input and target spatial sizes ≤256 via the disk cache; scale architectures down if compute becomes a bottleneck.

### Priority order (ease × leverage)
1. Composite reconstruction loss rollout (add SSC-aligned terms sequentially: SAM → SID → ΔE00).  
   _Status:_ Step 1A (L1 + λ·SAM) implemented — awaiting training run.
2. sRGB-rendered perceptual branch (linear sRGB L1 plus SSIM on gamma-corrected sRGB) to reinforce spatial/color fidelity.  
   _Status:_ queued.
3. Spectral smoothness prior (second-derivative TV along wavelength) to suppress ringing from coarse-to-fine spectral upsampling.  
   _Status:_ queued.
4. Augmentation refresh (90° rotations, diagonal flips, mild mosaic noise, spectral gain jitter) to expand effective dataset size.  
   _Status:_ queued.
5. Random crop + multi-scale schedule (mix 96–256 crops with periodic full frames) to trade compute for additional stochasticity.  
   _Status:_ queued.
6. Optimizer polish (cosine LR with warmup, gradient accumulation for effective larger batch) to stabilize longer runs.  
   _Status:_ Step 6A (cosine warmup scheduler) implemented; gradient accumulation still queued.
7. Training-time SSC metrics (reuse evaluator during training epochs) for faster feedback on SAM/SID/ΔE00 trends.  
   _Status:_ queued.
8. EMA weights for evaluation to smooth validation variance when experimenting with noisy losses.  
   _Status:_ queued.
9. Pixel-unshuffle RGGB unpacking prior to convolutions for immediate CFA awareness at negligible cost.  
   _Status:_ default in baseline (always on); evaluation confirms compatibility.
10. Residual UNet-lite backbone (3-level, 32→128 channels) with skip connections for broader receptive field while remaining CPU friendly.  
    _Status:_ queued.
11. Spectral refinement head (1×1 + depthwise 1D conv stacks) to replace linear interpolation and model inter-band structure explicitly.  
    _Status:_ queued.
12. Lightweight spectral attention (SE or frequency channel attention) in bottleneck blocks to adaptively weight bands.  
    _Status:_ queued.
13. Two-stage coarse-to-fine refinement (auxiliary residual head) for high-frequency cleanup once previous steps converge.  
    _Status:_ queued.

### Parking lot
- Record any new ideas or external references here before re-ranking them against the backlog.

---

## 15) Architecture upgrade plan

### Current baseline (post pixel-unshuffle)
- Mosaic input (1×H×W) → pixel-unshuffle (factor 2) → 4-channel tensor.
- Trunk: three 5×5 conv+ReLU blocks at 32 channels followed by 1×1 conv producing 28-channel coarse spectra (7 bands × 2² packing).
- Pixel shuffle restores spatial resolution, giving 7-band coarse spectra; linear interpolation upsamples 7→61 bands.
- No skip connections, limited receptive field (~13×13), parameter count ≈50k.
- Output is clamped to [0,1]; no spectral refinement beyond interpolation.

### Architecture backlog (ordered by ease × impact)
1. **UNet-lite backbone**  
   - 3 down / 3 up levels, base width 32→64→128.  
   - Residual double-conv blocks with 3×3 kernels; bilinear upsample + 1×1 to fuse skip connections.  
   - Directly predict 61 bands (drop interpolation) while keeping pixel-unshuffle stem.  
   - Target <2× baseline inference time on CPU for 256².
2. **Spectral refinement head**  
   - Append grouped 1×1 + depthwise separable conv stack along spectral axis to model inter-band structure, optionally predicting residual over coarse UNet output.
3. **PCA + residual output**  
   - Predict top-k PCA coefficients (precomputed basis) plus residual cube; reduces channel burden and encourages smooth spectra.
4. **Spectral attention**  
   - Insert squeeze-excite or frequency channel attention at bottleneck layers to adaptively weight bands.
5. **Two-stage refinement**  
   - Stage 1 coarse UNet-lite; Stage 2 shallow residual CNN that refines high-frequency detail using concatenated mosaic + stage1 prediction.

### Evaluation criteria for architecture experiments
- Metrics: SAM↓, SID↓, ΔE00↓ primary; ERGAS↓, PSNR_sRGB↑, SSIM_sRGB↑ secondary.  
- Runtime: record forward-pass wall-clock on CPU for 64² and 256² inputs, keeping <2× baseline.  
- Memory: ensure peak <~2 GB during training (batch 4 at 64²).  
- Stability: watch for NaNs; enable gradient clipping if necessary.

### Stage 1 implementation target — `unet_lite`
- Add config knob `--model-variant {baseline, unet_lite, ...}` with default `baseline`.
- UNet-lite specifics:  
  - Encoder levels at 32/64/128 channels with residual blocks and stride-2 conv for downsampling.  
  - Decoder uses bilinear upsampling + concat skip + residual block.  
  - Final 1×1 conv outputs 61 channels; apply `torch.sigmoid` or clamp for [0,1].  
  - Maintain pixel-unshuffle/shuffle at I/O boundaries.
- Implementation checklist:  
  1. Extend `TrainConfig`/CLI to accept `model_variant` (default `baseline`).  
  2. Add `UNetLiteHSI` module under `src/models/` (new file or alongside baseline) with parameterised base width.  
  3. Refactor model factory (in `train.py`/`evaluate.py`) to instantiate the requested variant and log parameter count.  
  4. Update checkpoint naming or metadata if variant differs to avoid accidental reload.  
  5. Unit smoke test via `python -m src.train --epochs 1 --batch-size 1 --resize-to 64 --model-variant unet_lite` to confirm forward/backward compatibility before long runs.
- Post-implementation experiment:  
  - Train `python -m src.train --data-root data/track1 --run-name unet-lite-v1 --epochs 40 --batch-size 4 --resize-to 64 --cache-dir data/cache/track1 --model-variant unet_lite`  
  - Evaluate `python -m src.evaluate --data-root data/track1 --run-name unet-lite-v1 --cache-dir data/cache/track1 --metrics sam,sid,ergas,psnr_srgb,ssim_srgb,deltae00`  
  - Also run native-resolution evaluation and capture forward-pass timing for 64² / 256² inputs.

### Logistics
- Keep legacy checkpoints compatible by gating new modules behind the variant flag; default training remains baseline until new model validated.
- Update training logs to print chosen variant and parameter count for reproducibility.
- If runtime exceeds limits, expose `--unet-base-channels` to downscale widths quickly.

### Loss extension — ERGAS term
- Implementation: ERGAS loss matches the evaluation metric and is now available via `--lambda-ergas`; the loss value is normalized (divided by 100) so weights around 0.05–0.1 are sensible. Defaults to 0 so existing runs are unaffected.
- Recommended sweep: start with λ_ergas=0.05 alongside the SAM/SID configuration; increase only if ERGAS remains high after tuning.
- Workflow:
  1. Baseline (no ERGAS):  
     `python -m src.train --run-name unet-lite-v1 --epochs 40 --batch-size 4 --resize-to 64 --model-variant unet_lite --lambda-l1 1.0 --lambda-sam 0.1 --lambda-sid 0.1 --lambda-srgb-l1 0.0 --lambda-srgb-ssim 0.0`
  2. ERGAS-enabled:  
     add `--lambda-ergas 0.05` (and adjust as needed).  
     Evaluate both runs with `python -m src.evaluate --run-name <run> --model-variant unet_lite --cache-dir data/cache/track1 --metrics sam,sid,ergas,psnr_srgb,ssim_srgb,deltae00`.

---

## 16) Experiment log (append entries sequentially)

| Stage | Idea | Run ID | Key settings | Val SAM | Val SID | Val ERGAS | Val PSNR_sRGB | Val SSIM_sRGB | Val ΔE00 | Notes | Status |
|-------|------|--------|--------------|---------|---------|-----------|---------------|---------------|----------|-------|--------|
| 0 | Baseline (L1 only) | demo-run (or user-provided) | bs=4, epochs=?, resize=64 | – | – | – | – | – | – | Fill with existing reference metrics if available. | queued |
| 1A | L1 + λ_SAM·SAM | loss-sam-v1 | λ_L1=1.0, λ_SAM=0.1, resize=64 | **14.44** | **0.064** | **66.60** | **25.05** | **0.934** | **6.72** | Stable with smooth SAM (1−cos). Improves SAM + SID on 64² while leaving RGB metrics roughly flat; native-res SAM drops to 18.42 with ΔE00 ≈8.27. | completed |
| 1B | + λ_SID·SID | loss-sam-sid-v1 | λ_L1=1.0, λ_SAM=0.1, λ_SID=0.1, resize=64 | **14.53** | **0.065** | 68.94 | 24.90 | **0.936** | **5.99** | λ_SID=0.1 improves SID and ΔE00 at 64²; native SAM 17.91, SID 0.112, ΔE00 7.00. Heavier weights (≥0.5) hurt spatial metrics without further SID gains. | completed |
| 1C | ΔE00 term (skipped) | – | – | – | – | – | – | Persistent instability (NaNs) despite clamps/eps tweaks; excluded from loss. | skipped |
| 1B.1 | L1 + λ_SAM (0.5) + λ_SID (0.2) | loss-sam-sid-v2 | λ `{L1:1.0, SAM:0.5, SID:0.2}`, resize=64 | **7.8225** | **0.0283** | 60.9271 | 24.5032 | 0.8980 | **4.6220** | Best 64² metrics to date; native eval SAM 15.39, SID 0.137, ERGAS 94.22, ΔE00 6.40. Spatial scores dip slightly; consider sRGB branch/TV next. | completed |
| 1D | + λ_ERGAS·ERGAS | loss-ergas-v1 | λ `{L1:1.0, SAM:0.1, SID:0.1, ERGAS:0.05}`, resize=64 | – | – | – | – | – | – | Compare against baseline to assess ΔE00/PSNR trade-offs; log both 64² and native evaluations. | pending_eval |
| 2 | Composite + sRGB branch | loss-srgb-v1 | λ_L1=1.0, λ_SAM=0.1, λ_SID=0.1, λ_srgb_L1=0.2, λ_srgb_SSIM=0.05 | – | – | – | – | – | – | Implementation landed; sweep pending. | pending_eval |
| 3 | Composite + TV tuning | TBD | λ_TV sweep | – | – | – | – | – | – | Pending. | queued |
| 4 | Pixel-unshuffle stem (default) | unshuffle-v1 | 20 epochs, bs=4, resize=64, λ `{L1:1.0, SAM:0.1, SID:0.1}` | **15.86** | **0.0827** | 80.28 | ~23.0† | 0.8789 | 8.0964 | CFA-aware stem active; evaluation without matching architecture raises size-mismatch (expected). †PSNR line truncated in console; per-final batch snapshot ~23 dB—confirm with rerun if needed. | completed |
| 5 | UNet-lite backbone | unet-lite-v1 | λ `{L1:1.0, SAM:0.1, SID:0.1, sRGB_L1:0.2, sRGB_SSIM:0.05}`, model_variant=unet_lite | – | – | – | – | – | – | Implementation landed; run pending. | pending_eval |
| 6A | Cosine warmup scheduler | sched-cosine-v1 | `lr_scheduler=cosine`, warmup=10, start_factor=0.2, eta_min=3e-5, resize=64, bs=1 | – | – | – | – | – | – | Scheduler integrated in codebase; run pending to compare against Step 1B. | pending_eval |
| 7 | UNet-lite native 1024 | unet-lite-native-v1 | model_variant=unet_lite, unet_base_channels=64, latent_channels=32, encoder_depth=8, coarse_channels=7 (default), resize=None (native), lr=1e-3, lr_scheduler=cosine (warmup_epochs=5, start_factor=0.2, eta_min=1e-5), λ `{L1:1.0, SAM:0.2, SID:0.1, sRGB_L1:0.2, sRGB_SSIM:0.05}`, bs=2, epochs=120, ram_cache=on, num_workers=1, prefetch_factor=4, resume | **13.8433** | **0.0604** | **23.5772** | **35.0968** | **0.9695** | **3.5426** | Eval (best ep 120): loss=0.0141. Kaggle test score ≈0.25 (SSC-style accuracy). Serves as native-resolution UNet-lite baseline. | completed |
| 8 | UNet-lite @256 resize | unet-lite-res256-v1 | model_variant=unet_lite, unet_base_channels=64, latent_channels=32, encoder_depth=7, coarse_channels=11, resize=256 (train+val), lr=1e-3, lr_scheduler=cosine (warmup_epochs=5, start_factor=0.2, eta_min=1e-5), λ `{L1:1.0, SAM:0.2, SID:0.1, sRGB_L1:0.2, sRGB_SSIM:0.05}`, bs=2, epochs=120, ram_cache=on, num_workers=1, prefetch_factor=4 | **13.4152** | **0.0577** | **22.7848** | **35.7559** | **0.9594** | **1.9976** | Eval (best ep 120): loss=0.0142. Kaggle test score ≈0.233. Spectral/color metrics improve vs native but SSIM_sRGB drops; suggests some trade-off between global spectral fidelity and fine spatial structure under the challenge metric. | completed |
| 9 | L1 + SAM curriculum @128² | unet-lite-res128-v3-l1sam | resize=128, bs=2, UNet-lite (base 64), λ `{L1:1.0, SAM:0.2}`, cosine LR (3e-4 → 1e-5), fine-tuned from `unet-lite-res128-v0` | **13.6722** | **0.0608** | **28.5282** | **31.0464** | **0.9413** | **3.0047** | Two-step fine-tune: λ_SAM=0.05 pass already improved SAM (13.84) and ERGAS (28.47); bumping to 0.2 yielded the listed metrics without hurting PSNR/SSIM. Kaggle submission pending. | completed |
| 10 | All composite losses (aggressive weights) | unet-lite-res128-v4-all-loss | resize=128, bs=2, UNet-lite (base 64), λ `{L1:1.0, SAM:0.2, SID:0.2, sRGB_L1:0.2, sRGB_SSIM:0.2, ERGAS:0.2}`, lr=1e-3 cosine schedule | **13.8236** | **0.0620** | **29.3705** | **30.9755** | **0.9425** | **3.2705** | Training converged (best ep 95, loss=0.0173) but composite weights at 0.2 for every term slightly degraded SAM/SID/ERGAS relative to the λ_SAM-only baseline, indicating the aux losses need gentler weighting and staged introduction. | completed |
| 11 | Residual head refinement | unet-lite-res128-v5-reshead | resize=128, bs=2, UNet-lite (base 64) + `--use-residual-head`, λ `{L1:1.0, SAM:0.2}`, lr=1e-3 cosine (warmup 5, η_min=1e-5), 180 epochs | **7.9279** | **0.0326** | **26.3974** | **31.6792** | **0.9434** | **2.1193** | Fine-tuned from the λ_SAM baseline with the residual refinement head active; best epoch 160 (loss 0.0111). Delivers the largest SAM/SID improvement so far while preserving spatial/color metrics. | completed |
| 12 | Residual head + spectral conv | unet-lite-res128-v5_1-reshead-spec | Same as Stage 11 plus `--use-spectral-conv --spectral-conv-kernel-size 3` (per-pixel 1D smoothing) | **7.8901** | **0.0325** | **26.0315** | **31.7652** | **0.9432** | **2.0937** | Adds spectral 1D convolution atop the residual head. Slight further gains in SAM/ERGAS/ΔE00 (best epoch 179, loss 0.0109). Candidate for Kaggle submission. | completed |
| 13 | Native 1024 UNet-lite + residual head (coarse=11) | unet-lite-res512-v5-reshead | resize=1024, bs=2; `model_variant=unet_lite`; `unet_base_channels=64`; `latent_channels=32`; `encoder_depth=6`; `coarse_channels=11`; lr=1e-3 cosine (warmup 5, η_min=1e-5); λ `{L1:1.0, SAM:0.2, SID:0.0, sRGB_L1:0.0, sRGB_SSIM:0.0, ERGAS:0.0}`; `--use-residual-head`; epochs=180; resume | **7.5673** | **0.0302** | **11.9962** | **40.0959** | **0.9678** | **1.5505** | Kaggle accuracy 0.3254 (public). Per-band MAE shows a U-shape with larger error at spectrum ends (<≈440 nm, >≈930 nm). Thin vertical residual columns at band extremes likely column FPN in GT + model’s early downsampling. | completed |
| … | (extend as needed) | | | | | | | | | | |


---

## 17) Scheduler rollout and long-run plan

- **Implementation recap** – Added `--lr-scheduler`, `--scheduler-warmup-epochs`, `--scheduler-warmup-start-factor`, and `--scheduler-min-lr` flags. The default remains `none`; selecting `cosine` enables a linear warmup (clamped to ≥1 epoch below the total) followed by cosine decay down to `eta_min`.
- **Recommended defaults for short 64² sweeps** – `--lr-scheduler cosine --scheduler-warmup-epochs 5 --scheduler-warmup-start-factor 0.2 --scheduler-min-lr 3e-5` with base LR `3e-4`. This keeps the first epoch gentler (0.2×) and approaches the baseline LR by epoch 5.
- **Projected long-run (≈2.2–2.8 h on CPU)** – Increase spatial size to 128, bump batch size to 2 (with gradient accumulation disabled) and widen the UNet-lite base channels to 64. Empirically expect ~4.5–5.5 minutes per epoch; 24–28 epochs should land inside a 3 h window with headroom.
  - Sanity check after epoch 3: if epoch time >6 minutes, drop batch size to 1 or reduce base channels to 56.
- **Command template**

```bash
python -m src.train \
  --run-name unet-lite-sched-v1 \
  --epochs 28 \
  --batch-size 2 \
  --resize-to 128 \
  --model-variant unet_lite \
  --unet-base-channels 64 \
  --learning-rate 3e-4 \
  --lr-scheduler cosine \
  --scheduler-warmup-epochs 5 \
  --scheduler-warmup-start-factor 0.2 \
  --scheduler-min-lr 3e-5 \
  --lambda-l1 1.0 \
  --lambda-sam 0.2 \
  --lambda-sid 0.2 \
  --lambda-srgb-l1 0.0 \
  --lambda-srgb-ssim 0.0 \
  --lambda-ergas 0.0
```

> Run on the cached 128² data; if memory spikes, fall back to `--batch-size 1` or `--unet-base-channels 56`. After completion, evaluate with native resolution to log scheduler impact in Stage 6A.

