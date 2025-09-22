from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple

from utils.packing import pack_2x2, unpack_2x2


def _safe_blur(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    return cv2.blur(x, (k, k), borderType=cv2.BORDER_REFLECT)


def _sobel_mag(x: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag


def build_features_from_packed(packed4: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    packed4: (4, H2, W2) float32 in [0,1]
    Returns: (H2*W2, D) features and (H2, W2)
    """
    assert packed4.ndim == 3 and packed4.shape[0] == 4
    C, H2, W2 = packed4.shape

    # Base channels
    chs = [packed4[i] for i in range(4)]  # TL, TR, BL, BR

    feats = []
    # identity per-channel
    feats += chs

    # Local means (3x3 and 5x5)
    means3 = [_safe_blur(c, 3) for c in chs]
    means5 = [_safe_blur(c, 5) for c in chs]
    feats += means3
    feats += means5

    # Local variance (3x3)
    sq = [c * c for c in chs]
    mean_sq3 = [_safe_blur(s, 3) for s in sq]
    var3 = [np.maximum(0.0, ms - m * m) for ms, m in zip(mean_sq3, means3)]
    feats += var3

    # Sobel magnitude per channel
    sob = [_sobel_mag(c) for c in chs]
    feats += sob

    # Stack and reshape to (H2*W2, D)
    F = np.stack(feats, axis=0).astype(np.float32)
    D = F.shape[0]
    X = F.reshape(D, -1).T
    return X, (H2, W2)


def pack_cube_target(cube_chw: np.ndarray) -> np.ndarray:
    """cube_chw: (61, H, W) -> target (61*4, H/2, W/2)"""
    return pack_2x2(cube_chw)


def unpack_pred_to_cube(pred_packed: np.ndarray, bands: int = 61) -> np.ndarray:
    """pred_packed: (bands*4, H2, W2) -> (bands, H, W)"""
    return unpack_2x2(pred_packed, bands)

