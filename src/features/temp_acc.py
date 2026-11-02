"""Temperature (TEMP) and Accelerometer (ACC) feature extraction and motion quality gating.

Features:
  - TEMP: mean, std, linear slope (deg C/sec)
  - ACC: magnitude mean, std, spectral energy
  - Quality: high-motion fraction (fraction of window with significant movement)
"""

from __future__ import annotations

from typing import Dict
import numpy as np


def extract_temp_features(temp_raw: np.ndarray, fs_hz: float = 4.0) -> Dict[str, float]:
    """Extract skin temperature features: mean, std, linear slope."""
    arr = np.asarray(temp_raw, dtype=np.float64).flatten()
    if len(arr) == 0:
        return {"temp_mean": 32.0, "temp_std": 0.0, "temp_slope": 0.0}

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))

    # Slope in deg C / sec
    if len(arr) >= 2:
        x = np.arange(len(arr)) / fs_hz
        x_mean = np.mean(x)
        slope = float(np.sum((x - x_mean) * (arr - mean_val)) / np.sum((x - x_mean) ** 2))
    else:
        slope = 0.0

    return {
        "temp_mean": mean_val,
        "temp_std": std_val,
        "temp_slope": slope,
    }


def extract_acc_features(acc_raw: np.ndarray, fs_hz: float = 32.0, high_motion_threshold_g: float = 0.3) -> Dict[str, float]:
    """Extract 3-axis accelerometer features and motion quality metric.
    
    Args:
        acc_raw: Shape (N, 3) in units of standard g.
        fs_hz: Accelerometer sample rate (32 Hz).
        high_motion_threshold_g: Threshold for dynamic movement above gravity baseline.
        
    Returns:
        Dict with magnitude mean, std, energy, and high_motion_fraction.
    """
    arr = np.asarray(acc_raw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3) if len(arr) % 3 == 0 else arr.reshape(-1, 1)

    if len(arr) == 0:
        return {
            "acc_mag_mean": 1.0,
            "acc_mag_std": 0.0,
            "acc_energy": 1.0,
            "high_motion_fraction": 0.0,
        }

    # 3-axis magnitude
    if arr.shape[1] >= 3:
        mag = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    else:
        mag = np.abs(arr[:, 0])

    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))
    # Energy: average squared magnitude
    energy = float(np.mean(mag ** 2))

    # Dynamic acceleration: deviation from 1.0 g gravitational baseline
    dynamic_acc = np.abs(mag - 1.0)
    high_motion_samples = np.sum(dynamic_acc > high_motion_threshold_g)
    high_motion_fraction = float(high_motion_samples / len(mag))

    return {
        "acc_mag_mean": mag_mean,
        "acc_mag_std": mag_std,
        "acc_energy": energy,
        "high_motion_fraction": high_motion_fraction,
    }
