"""Electrodermal Activity (EDA) feature extraction: tonic and phasic decomposition.

Extracts:
  - Statistical: mean, std, linear slope
  - Tonic: estimated mean tonic level (SCL)
  - Phasic: skin conductance response (SCR) peak count and amplitude sum
"""

from __future__ import annotations

from typing import Dict
import numpy as np
from scipy import signal


def decompose_eda(eda_raw: np.ndarray, fs_hz: float = 4.0, tonic_cutoff_hz: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Decompose raw EDA into tonic (slow) and phasic (fast) components.
    
    Uses zero-phase 2nd order Butterworth low-pass filter for the tonic baseline.
    """
    eda_arr = np.asarray(eda_raw, dtype=np.float64).flatten()
    if len(eda_arr) < 12:
        return eda_arr, np.zeros_like(eda_arr)

    nyq = 0.5 * fs_hz
    cutoff = min(tonic_cutoff_hz / nyq, 0.99)
    b, a = signal.butter(2, cutoff, btype="low")
    tonic = signal.filtfilt(b, a, eda_arr)
    phasic = eda_arr - tonic
    # Phasic responses are non-negative skin conductance responses
    phasic = np.maximum(0.0, phasic)
    return tonic, phasic


def detect_scr_peaks(phasic: np.ndarray, fs_hz: float = 4.0, min_amplitude: float = 0.02) -> tuple[int, float]:
    """Detect phasic Skin Conductance Responses (SCR).
    
    Standard threshold: minimum amplitude of 0.02 uS, minimum separation 1.0s.
    """
    min_dist = max(1, int(fs_hz * 1.0))
    peaks, props = signal.find_peaks(phasic, height=min_amplitude, distance=min_dist)
    peak_count = len(peaks)
    amplitude_sum = float(np.sum(props["peak_heights"])) if peak_count > 0 else 0.0
    return peak_count, amplitude_sum


def compute_linear_slope(y: np.ndarray, fs_hz: float = 4.0) -> float:
    """Compute slope (rate of change per second) using simple linear regression."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y)) / fs_hz
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return float(numerator / denominator) if denominator != 0 else 0.0


def extract_eda_features(eda_raw: np.ndarray, fs_hz: float = 4.0) -> Dict[str, float]:
    """Complete EDA feature pipeline."""
    eda_arr = np.asarray(eda_raw, dtype=np.float64).flatten()
    if len(eda_arr) == 0:
        return {
            "eda_mean": 0.0,
            "eda_std": 0.0,
            "eda_slope": 0.0,
            "eda_tonic_level": 0.0,
            "eda_phasic_peaks": 0.0,
            "eda_phasic_amplitude_sum": 0.0,
        }

    tonic, phasic = decompose_eda(eda_arr, fs_hz=fs_hz)
    peak_count, amp_sum = detect_scr_peaks(phasic, fs_hz=fs_hz)
    slope = compute_linear_slope(eda_arr, fs_hz=fs_hz)

    return {
        "eda_mean": float(np.mean(eda_arr)),
        "eda_std": float(np.std(eda_arr)),
        "eda_slope": slope,
        "eda_tonic_level": float(np.mean(tonic)),
        "eda_phasic_peaks": float(peak_count),
        "eda_phasic_amplitude_sum": amp_sum,
    }
