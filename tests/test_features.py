"""Unit tests for Phase 1C: Feature extraction, quality metrics, normalizers, and CNN tensors."""

import unittest
import numpy as np
import pandas as pd
from src.features.bvp import filter_bvp, detect_bvp_peaks, compute_hr_hrv_features, extract_bvp_features
from src.features.eda import decompose_eda, detect_scr_peaks, compute_linear_slope, extract_eda_features
from src.features.temp_acc import extract_temp_features, extract_acc_features
from src.features.tensor_builder import build_window_tensor, resample_bvp_to_32hz, interpolate_to_32hz
from src.features.normalizer import GlobalZScoreNormalizer, PerSubjectBaselineNormalizer
from src.features.extractor import extract_window_features_and_tensor


class TestFeaturesAndTensors(unittest.TestCase):
    def test_bvp_hr_and_hrv_extraction(self):
        """Test BVP filtering, peak detection, and HR/HRV calculation on a simulated 1 Hz (60 bpm) pulse."""
        fs = 64.0
        duration = 10.0 # 10 seconds
        t = np.arange(0, duration, 1.0 / fs)
        # 1 Hz pulse with harmonic to mimic PPG waveform
        bvp_sig = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 2.0 * t)

        feats = extract_bvp_features(bvp_sig, fs_hz=fs)
        # Expected HR ~ 60 bpm
        self.assertAlmostEqual(feats["hr_mean"], 60.0, delta=5.0)
        self.assertIn("hrv_rmssd", feats)
        self.assertIn("hrv_sdnn", feats)
        self.assertIn("hrv_pnn50", feats)
        self.assertIn("fraction_invalid_bvp", feats)

    def test_eda_decomposition_and_scr(self):
        """Test EDA decomposition into tonic baseline and phasic SCR peaks."""
        fs = 4.0
        duration = 60.0
        t = np.arange(0, duration, 1.0 / fs)
        # Constant tonic baseline of 2 uS with 2 distinct phasic SCR pulses
        eda_sig = np.full_like(t, 2.0)
        # Add SCR peaks at t=10s and t=30s
        eda_sig += 0.5 * np.exp(-((t - 10.0) ** 2) / 2.0)
        eda_sig += 0.8 * np.exp(-((t - 30.0) ** 2) / 2.0)

        feats = extract_eda_features(eda_sig, fs_hz=fs)
        self.assertGreaterEqual(feats["eda_phasic_peaks"], 1)
        self.assertGreater(feats["eda_phasic_amplitude_sum"], 0.0)
        self.assertAlmostEqual(feats["eda_tonic_level"], 2.0, delta=0.5)

    def test_temp_and_acc_features(self):
        """Test temperature slope and accelerometer magnitude, energy, and motion gating."""
        # 1. TEMP: upward drift of 1 deg over 10s (slope = 0.1 deg/s)
        temp_sig = np.linspace(30.0, 31.0, 40) # 4 Hz, 10s
        temp_feats = extract_temp_features(temp_sig, fs_hz=4.0)
        self.assertAlmostEqual(temp_feats["temp_mean"], 30.5, delta=0.1)
        self.assertAlmostEqual(temp_feats["temp_slope"], 0.1, delta=0.01)

        # 2. ACC: stationary wrist at 1g along z-axis
        acc_sig = np.zeros((320, 3)) # 32 Hz, 10s
        acc_sig[:, 2] = 1.0 # 1g
        acc_feats = extract_acc_features(acc_sig, fs_hz=32.0)
        self.assertAlmostEqual(acc_feats["acc_mag_mean"], 1.0, delta=0.01)
        self.assertAlmostEqual(acc_feats["acc_energy"], 1.0, delta=0.01)
        self.assertEqual(acc_feats["high_motion_fraction"], 0.0)

    def test_cnn_tensor_builder_shape_and_channels(self):
        """Test resampling to 32 Hz and tensor shape of (1920, 6) for 60s windows."""
        expected_len = 1920
        bvp_64hz = np.ones(3840, dtype=np.float32)
        eda_4hz = np.full(240, 2.0, dtype=np.float32)
        temp_4hz = np.full(240, 32.0, dtype=np.float32)
        acc_32hz = np.zeros((1920, 3), dtype=np.float32)
        acc_32hz[:, 2] = 1.0

        tensor = build_window_tensor(bvp_64hz, eda_4hz, temp_4hz, acc_32hz, expected_length=expected_len)
        self.assertEqual(tensor.shape, (expected_len, 6))
        # Channel 0: BVP
        self.assertAlmostEqual(float(tensor[0, 0]), 1.0, delta=0.1)
        # Channel 1: EDA
        self.assertAlmostEqual(float(tensor[0, 1]), 2.0, delta=0.1)
        # Channel 2: TEMP
        self.assertAlmostEqual(float(tensor[0, 2]), 32.0, delta=0.1)
        # Channel 5: ACC_z
        self.assertAlmostEqual(float(tensor[0, 5]), 1.0, delta=0.01)

    def test_global_zscore_isolation(self):
        """Test that normalizer fits solely on training fold without leaking test fold statistics."""
        train_df = pd.DataFrame({"feat_a": [10.0, 20.0, 30.0], "label": [0, 1, 0]})
        test_df = pd.DataFrame({"feat_a": [100.0, 200.0], "label": [1, 1]})

        norm = GlobalZScoreNormalizer(feature_cols=["feat_a"])
        # Unfitted transform must raise
        with self.assertRaises(RuntimeError):
            norm.transform(test_df)

        norm.fit(train_df)
        self.assertAlmostEqual(norm.means_["feat_a"], 20.0)
        # Test transform uses train mean (20.0)
        test_trans = norm.transform(test_df)
        self.assertAlmostEqual(test_trans["feat_a"].iloc[0], (100.0 - 20.0) / norm.stds_["feat_a"])

    def test_master_extractor(self):
        """Test master extractor returns both tabular dictionary and CNN tensor."""
        bvp = np.sin(np.linspace(0, 20 * np.pi, 3840))
        eda = np.linspace(1.0, 1.5, 240)
        temp = np.linspace(31.0, 32.0, 240)
        acc = np.zeros((1920, 3))
        acc[:, 2] = 1.0

        feats, tensor = extract_window_features_and_tensor(bvp, eda, temp, acc, expected_tensor_length=1920)
        self.assertIn("hr_mean", feats)
        self.assertIn("eda_tonic_level", feats)
        self.assertIn("temp_mean", feats)
        self.assertIn("acc_energy", feats)
        self.assertIn("fraction_invalid_bvp", feats)
        self.assertIn("high_motion_fraction", feats)
        self.assertEqual(tensor.shape, (1920, 6))


if __name__ == "__main__":
    unittest.main()
