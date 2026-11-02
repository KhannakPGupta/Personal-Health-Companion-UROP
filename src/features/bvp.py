"""BVP signal processing, peak detection, HR, HRV, and quality assessment.

Features extracted strictly from raw BVP (without using vendor HR.csv):
  - 0.5 to 8.0 Hz bandpass filtering (Butterworth)
  - Systolic peak detection
  - HR metrics: mean, std, min, max (bpm)
  - HRV metrics: RMSSD, SDNN, pNN50
  - Signal quality: fraction of invalid beats
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from scipy import signal


def filter_bvp(bvp_raw: np.ndarray, fs_hz: float = 64.0, lowcut: float = 0.5, highcut: float = 8.0, order: int = 3) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter to raw BVP.
    
    Args:
        bvp_raw: 1D raw BVP signal array.
        fs_hz: Sampling frequency in Hz (default: 64.0).
        lowcut: Lower cutoff frequency (default: 0.5 Hz).
        highcut: Upper cutoff frequency (default: 8.0 Hz).
        order: Filter order.
        
    Returns:
        Filtered BVP signal.
    """
    bvp_clean = np.asarray(bvp_raw, dtype=np.float64).flatten()
    if len(bvp_clean) < order * 3:
        return bvp_clean

    nyquist = 0.5 * fs_hz
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.99)
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return signal.filtfilt(b, a, bvp_clean)


def detect_bvp_peaks(bvp_filtered: np.ndarray, fs_hz: float = 64.0) -> np.ndarray:
    """Detect systolic peaks from bandpass-filtered BVP signal.
    
    Minimum distance corresponds to 220 bpm (fs * 60 / 220 = ~0.27s).
    """
    min_distance = int(fs_hz * 0.27)
    # Adaptive prominence based on standard deviation
    std_val = np.std(bvp_filtered)
    prominence = max(0.1 * std_val, 1e-4)

    peaks, _ = signal.find_peaks(bvp_filtered, distance=min_distance, prominence=prominence)
    return peaks


def compute_hr_hrv_features(
    peaks: np.ndarray,
    fs_hz: float = 64.0,
    min_valid_ibi_ms: float = 300.0,
    max_valid_ibi_ms: float = 2000.0,
) -> Tuple[Dict[str, float], float]:
    """Calculate HR and HRV metrics from detected peak indices.
    
    Args:
        peaks: Indices of detected systolic peaks.
        fs_hz: Sampling frequency.
        min_valid_ibi_ms: Minimum physiological IBI (300 ms = 200 bpm).
        max_valid_ibi_ms: Maximum physiological IBI (2000 ms = 30 bpm).
        
    Returns:
        Tuple of (feature_dict, fraction_invalid_beats).
    """
    default_metrics = {
        "hr_mean": 70.0,
        "hr_std": 0.0,
        "hr_min": 70.0,
        "hr_max": 70.0,
        "hrv_rmssd": 0.0,
        "hrv_sdnn": 0.0,
        "hrv_pnn50": 0.0,
    }

    if len(peaks) < 3:
        return default_metrics, 1.0  # 100% invalid if fewer than 3 peaks detected

    # Inter-beat intervals in milliseconds
    ibi_ms = np.diff(peaks) / fs_hz * 1000.0
    total_beats = len(ibi_ms)

    # Valid physiological window [300ms, 2000ms]
    valid_mask = (ibi_ms >= min_valid_ibi_ms) & (ibi_ms <= max_valid_ibi_ms)

    # Relative jump rejection: IBI change > 30% from preceding beat is physiologically suspect in resting/steady states
    if np.sum(valid_mask) > 2:
        diff_successive = np.abs(np.diff(ibi_ms))
        diff_mask = np.ones(len(ibi_ms), dtype=bool)
        diff_mask[1:] = diff_successive <= (0.4 * ibi_ms[:-1])
        valid_mask = valid_mask & diff_mask

    invalid_count = total_beats - np.sum(valid_mask)
    fraction_invalid = float(invalid_count / total_beats)

    clean_ibi = ibi_ms[valid_mask]
    if len(clean_ibi) < 2:
        return default_metrics, fraction_invalid

    # Heart rate in BPM
    hr_bpm = 60000.0 / clean_ibi

    # HRV: SDNN
    sdnn = float(np.std(clean_ibi, ddof=1)) if len(clean_ibi) > 1 else 0.0

    # HRV: RMSSD and pNN50 on successive clean IBIs
    successive_diffs = np.diff(clean_ibi)
    if len(successive_diffs) > 0:
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
        pnn50 = float(np.mean(np.abs(successive_diffs) > 50.0) * 100.0)
    else:
        rmssd = 0.0
        pnn50 = 0.0

    features = {
        "hr_mean": float(np.mean(hr_bpm)),
        "hr_std": float(np.std(hr_bpm)),
        "hr_min": float(np.min(hr_bpm)),
        "hr_max": float(np.max(hr_bpm)),
        "hrv_rmssd": rmssd,
        "hrv_sdnn": sdnn,
        "hrv_pnn50": pnn50,
    }
    return features, fraction_invalid


def extract_bvp_features(bvp_raw: np.ndarray, fs_hz: float = 64.0) -> Dict[str, float]:
    """Complete BVP pipeline: filtering -> peak detection -> HR/HRV/quality features."""
    filtered = filter_bvp(bvp_raw, fs_hz=fs_hz)
    peaks = detect_bvp_peaks(filtered, fs_hz=fs_hz)
    features, fraction_invalid = compute_hr_hrv_features(peaks, fs_hz=fs_hz)
    features["fraction_invalid_bvp"] = fraction_invalid
    return features
