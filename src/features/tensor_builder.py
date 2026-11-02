"""Raw-window tensor builder for 1D-CNN (M2).

Resamples all wrist sensor channels to a common 32 Hz rate:
  - BVP: Decimated from 64 Hz to 32 Hz with anti-aliasing filter (factor 2).
  - EDA: Linearly interpolated from 4 Hz to 32 Hz (factor 8).
  - TEMP: Linearly interpolated from 4 Hz to 32 Hz (factor 8).
  - ACC: Retained at native 32 Hz (x, y, z axes).
  - Output channel order: [BVP, EDA, TEMP, ACC_x, ACC_y, ACC_z] (6 channels, length 1920 for 60s).
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def resample_bvp_to_32hz(bvp_64hz: np.ndarray) -> np.ndarray:
    """Decimate BVP from 64 Hz to 32 Hz using an anti-aliasing filter (factor of 2)."""
    arr = np.asarray(bvp_64hz, dtype=np.float32).flatten()
    if len(arr) < 16:
        # Fallback if window is very short
        target_len = max(1, len(arr) // 2)
        return signal.resample(arr, target_len).astype(np.float32)
    # scipy.signal.decimate applies an 8th-order Chebyshev type I anti-aliasing filter
    return signal.decimate(arr, q=2, ftype="iir", zero_phase=True).astype(np.float32)


def interpolate_to_32hz(data_4hz: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly interpolate 4 Hz signal (EDA or TEMP) to 32 Hz matching target_length."""
    arr = np.asarray(data_4hz, dtype=np.float32).flatten()
    if len(arr) == 0:
        return np.zeros(target_length, dtype=np.float32)
    if len(arr) == 1:
        return np.full(target_length, arr[0], dtype=np.float32)

    x_orig = np.linspace(0.0, 1.0, len(arr))
    x_target = np.linspace(0.0, 1.0, target_length)
    return np.interp(x_target, x_orig, arr).astype(np.float32)


def build_window_tensor(
    bvp_64hz: np.ndarray,
    eda_4hz: np.ndarray,
    temp_4hz: np.ndarray,
    acc_32hz: np.ndarray,
    expected_length: int = 1920,
) -> np.ndarray:
    """Construct a 6-channel 32 Hz tensor of shape (expected_length, 6).
    
    Args:
        bvp_64hz: Raw BVP samples (~3840 for 60s).
        eda_4hz: Raw EDA samples (~240 for 60s).
        temp_4hz: Raw TEMP samples (~240 for 60s).
        acc_32hz: Raw ACC samples (~1920, 3 for 60s).
        expected_length: Target sequence length (1920 for 60s @ 32Hz).
        
    Returns:
        Numpy array of shape (expected_length, 6).
    """
    # 1. BVP (decimate to 32Hz)
    bvp_32 = resample_bvp_to_32hz(bvp_64hz)
    if len(bvp_32) > expected_length:
        bvp_32 = bvp_32[:expected_length]
    elif len(bvp_32) < expected_length:
        bvp_32 = np.pad(bvp_32, (0, expected_length - len(bvp_32)), mode="edge")

    # 2. EDA (interpolate to 32Hz)
    eda_32 = interpolate_to_32hz(eda_4hz, target_length=expected_length)

    # 3. TEMP (interpolate to 32Hz)
    temp_32 = interpolate_to_32hz(temp_4hz, target_length=expected_length)

    # 4. ACC (native 32Hz, 3-axis)
    acc_arr = np.asarray(acc_32hz, dtype=np.float32)
    if acc_arr.ndim == 1:
        acc_arr = acc_arr.reshape(-1, 3) if len(acc_arr) % 3 == 0 else np.zeros((expected_length, 3), dtype=np.float32)

    if len(acc_arr) > expected_length:
        acc_32 = acc_arr[:expected_length, :3]
    elif len(acc_arr) < expected_length:
        pad_rows = expected_length - len(acc_arr)
        last_row = acc_arr[-1:, :3] if len(acc_arr) > 0 else np.zeros((1, 3), dtype=np.float32)
        acc_32 = np.vstack([acc_arr[:, :3], np.repeat(last_row, pad_rows, axis=0)])
    else:
        acc_32 = acc_arr[:, :3]

    # Stack into shape (expected_length, 6)
    tensor = np.column_stack([
        bvp_32,
        eda_32,
        temp_32,
        acc_32[:, 0],
        acc_32[:, 1],
        acc_32[:, 2],
    ]).astype(np.float32)

    return tensor
