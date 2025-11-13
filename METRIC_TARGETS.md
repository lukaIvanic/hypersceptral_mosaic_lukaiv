# Validation Metric Targets

Reference ranges for Track 1 public validation (native resolution) based on top‐tier systems and internal experience. Use these as stretch goals while monitoring overall SSC performance.

| Metric | Target Range | Notes |
| --- | --- | --- |
| SAM ↓ | ≤ **7°** | Primary spectral fidelity indicator; improvements here often lower SID and ΔE00. |
| SID ↓ | ≤ **0.03** | Complements SAM by penalising spectral distribution mismatches. |
| ΔE00 ↓ | ≤ **4.0** | Key colour component; <4 is near imperceptible differences. |
| ERGAS ↓ | ≤ **60** | Normalised spectral RMSE; strong models land in the 45–60 band. |
| PSNR_sRGB ↑ | ≥ **25 dB** | Sanity check for spatial fidelity; SSC less sensitive once ≥25. |
| SSIM_sRGB ↑ | ≥ **0.93** | Tracks spatial/detail preservation in rendered RGB. |
| L1 loss ↓ | ≈ **0.02** at 64² | For composite objectives, expect similar magnitude when SAM/SID tuned. |

_Reminder:_ SSC weighting isn’t published; prioritise reducing SAM/SID/ΔE00 first, then refine ERGAS and spatial metrics without regressing colour fidelity.

