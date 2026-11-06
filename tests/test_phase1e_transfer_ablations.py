"""Unit tests for Phase 1E Cross-Dataset Transfer (E2) and Sensor/Window Ablations (E3)."""

import unittest
import numpy as np
import pandas as pd

from src.eval.cross_dataset_transfer import (
    run_cross_dataset_tabular_transfer,
    run_cross_dataset_cnn_transfer,
)
from src.eval.ablations import (
    run_sensor_modality_ablation,
    run_window_length_ablation,
    run_normalization_ablation,
    run_amusement_sensitivity_ablation,
)
from src.eval.run_phase1_experiments import generate_synthetic_cohort, TABULAR_FEATURE_COLS


class TestPhase1ETransferAndAblations(unittest.TestCase):
    def setUp(self):
        self.df_wesad, self.ten_wesad = generate_synthetic_cohort(
            "WESAD", ["S2", "S3", "S4"], windows_per_subject=10, seq_len=128, seed=42
        )
        self.df_physio, self.ten_physio = generate_synthetic_cohort(
            "PhysioNet", ["P01", "P02", "P03"], windows_per_subject=10, seq_len=128, seed=101
        )

    def test_cross_dataset_tabular_transfer_bidirectional(self):
        """Test bidirectional tabular transfer (WESAD <-> PhysioNet) with source validation calibration."""
        res_w2p, sum_w2p = run_cross_dataset_tabular_transfer(
            self.df_wesad,
            self.df_physio,
            model_name="m0_logistic_regression",
            feature_cols=TABULAR_FEATURE_COLS,
            source_name="WESAD",
            target_name="PhysioNet",
            calibrate_on_source_val=True,
        )
        self.assertEqual(len(res_w2p), 3)
        self.assertEqual(sum_w2p["direction"], "WESAD->PhysioNet")
        self.assertIn("macro_f1_mean", sum_w2p)

        res_p2w, sum_p2w = run_cross_dataset_tabular_transfer(
            self.df_physio,
            self.df_wesad,
            model_name="m1_random_forest",
            feature_cols=TABULAR_FEATURE_COLS,
            source_name="PhysioNet",
            target_name="WESAD",
            calibrate_on_source_val=False,
        )
        self.assertEqual(len(res_p2w), 3)
        self.assertEqual(sum_p2w["direction"], "PhysioNet->WESAD")

    def test_cross_dataset_leakage_rejection(self):
        """Verify that overlapping subject IDs between source and target raise AssertionError."""
        df_leaky = self.df_physio.copy()
        df_leaky.loc[df_leaky.index[0], "subject_id"] = "S2"  # Collides with WESAD S2
        with self.assertRaises(AssertionError):
            run_cross_dataset_tabular_transfer(
                self.df_wesad,
                df_leaky,
                model_name="m0_logistic_regression",
                feature_cols=TABULAR_FEATURE_COLS,
            )

    def test_cross_dataset_cnn_transfer(self):
        """Test M2 1D-CNN cross-dataset transfer."""
        res_df, summary = run_cross_dataset_cnn_transfer(
            self.ten_wesad,
            self.df_wesad["label"].values,
            self.df_wesad["subject_id"].values,
            self.ten_physio,
            self.df_physio["label"].values,
            self.df_physio["subject_id"].values,
            source_name="WESAD",
            target_name="PhysioNet",
            epochs=3,
            batch_size=8,
            patience=2,
        )
        self.assertEqual(len(res_df), 3)
        self.assertEqual(summary["model"], "m2_cnn_1d")
        self.assertIn("macro_f1_95ci_lower", summary)

    def test_sensor_window_and_normalization_ablations(self):
        """Test sensor modality, window duration, normalization, and amusement ablations."""
        df_sensor = run_sensor_modality_ablation(
            self.df_wesad, self.ten_wesad, dataset_name="WESAD", include_cnn=True, cnn_epochs=2
        )
        self.assertGreaterEqual(len(df_sensor), 6)
        self.assertIn("BVP_ONLY", df_sensor["configuration"].values)
        self.assertIn("ALL_SENSORS", df_sensor["configuration"].values)

        df_wesad_30, ten_wesad_30 = generate_synthetic_cohort(
            "WESAD", ["S2", "S3", "S4"], windows_per_subject=12, seq_len=64, seed=99
        )
        df_win = run_window_length_ablation(
            self.df_wesad, self.ten_wesad, df_wesad_30, ten_wesad_30, dataset_name="WESAD", cnn_epochs=2
        )
        self.assertEqual(len(df_win), 4)

        df_norm = run_normalization_ablation(self.df_wesad, dataset_name="WESAD")
        self.assertEqual(len(df_norm), 2)

        df_amuse = run_amusement_sensitivity_ablation(self.df_wesad, dataset_name="WESAD")
        self.assertEqual(len(df_amuse), 2)


if __name__ == "__main__":
    unittest.main()
