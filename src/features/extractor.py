"""Master feature and tensor extractor for 60s windows across WESAD and PhysioNet.

Extracts identical features for both datasets strictly from raw BVP, EDA, TEMP, and ACC:
  - BVP HR & HRV: hr_mean, hr_std, hr_min, hr_max, hrv_rmssd, hrv_sdnn, hrv_pnn50
  - EDA: eda_mean, eda_std, eda_slope, eda_tonic_level, eda_phasic_peaks, eda_phasic_amplitude_sum
  - TEMP: temp_mean, temp_std, temp_slope
  - ACC: acc_mag_mean, acc_mag_std, acc_energy
  - Quality features: fraction_invalid_bvp, high_motion_fraction
  - CNN tensor: 6-channel 32 Hz resampled tensor (1920, 6)
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.features.bvp import extract_bvp_features
from src.features.eda import extract_eda_features
from src.features.temp_acc import extract_temp_features, extract_acc_features
from src.features.tensor_builder import build_window_tensor


def extract_window_features_and_tensor(
    bvp_64hz: np.ndarray,
    eda_4hz: np.ndarray,
    temp_4hz: np.ndarray,
    acc_32hz: np.ndarray,
    expected_tensor_length: int = 1920,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute identical tabular features and 32 Hz tensor for a single window.
    
    Args:
        bvp_64hz: Raw BVP slice (~3840 samples for 60s @ 64Hz).
        eda_4hz: Raw EDA slice (~240 samples for 60s @ 4Hz).
        temp_4hz: Raw TEMP slice (~240 samples for 60s @ 4Hz).
        acc_32hz: Raw ACC slice (~1920 samples for 60s @ 32Hz, in standard g).
        expected_tensor_length: Target length for CNN tensor (1920 for 60s @ 32Hz).
        
    Returns:
        Tuple of (feature_dict, cnn_tensor_array_shape_1920x6).
    """
    bvp_feats = extract_bvp_features(bvp_64hz, fs_hz=64.0)
    eda_feats = extract_eda_features(eda_4hz, fs_hz=4.0)
    temp_feats = extract_temp_features(temp_4hz, fs_hz=4.0)
    acc_feats = extract_acc_features(acc_32hz, fs_hz=32.0)

    # Combine tabular features
    features = {}
    features.update(bvp_feats)
    features.update(eda_feats)
    features.update(temp_feats)
    features.update(acc_feats)

    # Build 6-channel 32Hz tensor
    tensor = build_window_tensor(
        bvp_64hz, eda_4hz, temp_4hz, acc_32hz, expected_length=expected_tensor_length
    )

    return features, tensor
