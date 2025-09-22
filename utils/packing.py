from __future__ import annotations
import numpy as np


def _ensure_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    # If HWC
    if arr.shape[0] not in (1, 3, 4, 31, 61):
        return np.transpose(arr, (2, 0, 1))
    return arr


def pack_2x2(arr: np.ndarray) -> np.ndarray:
    """
    PixelUnshuffle(2) equivalent for NumPy arrays.
    CHW -> (C*4, H//2, W//2)
    Order: TL, TR, BL, BR
    """
    x = _ensure_chw(arr)
    C, H, W = x.shape
    H2, W2 = H // 2, W // 2
    out = np.zeros((C * 4, H2, W2), dtype=x.dtype)
    out[0*C:1*C] = x[:, 0::2, 0::2]
    out[1*C:2*C] = x[:, 0::2, 1::2]
    out[2*C:3*C] = x[:, 1::2, 0::2]
    out[3*C:4*C] = x[:, 1::2, 1::2]
    return out


def unpack_2x2(arr_packed: np.ndarray, C: int) -> np.ndarray:
    """
    PixelShuffle(2) equivalent.
    (C*4, H2, W2) -> (C, 2*H2, 2*W2)
    """
    C4, H2, W2 = arr_packed.shape
    assert C4 == C * 4, f"Expected first dim {C*4}, got {C4}"
    y = np.zeros((C, H2 * 2, W2 * 2), dtype=arr_packed.dtype)
    y[:, 0::2, 0::2] = arr_packed[0*C:1*C]
    y[:, 0::2, 1::2] = arr_packed[1*C:2*C]
    y[:, 1::2, 0::2] = arr_packed[2*C:3*C]
    y[:, 1::2, 1::2] = arr_packed[3*C:4*C]
    return y

