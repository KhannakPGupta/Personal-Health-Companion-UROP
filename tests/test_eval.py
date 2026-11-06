"""Unit tests for Phase 1D evaluation metrics, calibration, and LOSO cross-validation."""

import unittest
import numpy as np
import pandas as pd
from src.eval.metrics import (
    compute_classification_metrics,
    compute_expected_calibration_error,
    compute_false_alerts_per_hour,
    bootstrap_subject_confidence_intervals,
)
from src.eval.calibration import SubjectCalibrator, evaluate_selective_prediction
from src.eval.loso_evaluator import run_loso_evaluation, run_cnn_loso_evaluation


class TestEvaluationAndCalibration(unittest.TestCase):
    def test_metrics_calculation(self):
        """Test primary classification metrics on controlled binary predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7])

        metrics = compute_classification_metrics(y_true, y_prob)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["strain_f1"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertLess(metrics["brier_score"], 0.1)
        self.assertIn("ece", metrics)

    def test_false_alerts_per_hour_persistence_rule(self):
        """Test that 3-of-5 persistence rule suppresses isolated spurious false alarms."""
        # 120 baseline windows (1 hour of recording with 30s step: 120 * 30s = 3600s)
        y_true = np.zeros(120, dtype=int)
        y_pred = np.zeros(120, dtype=int)

        # Inject 2 sporadic false positives at indices 10 and 50 (never 3 in a row)
        y_pred[10] = 1
        y_pred[50] = 1

        # Raw false alert rate = 2 alerts / hour
        raw_fa = compute_false_alerts_per_hour(y_true, y_pred, window_step_sec=30.0, persistence_rule=False)
        self.assertAlmostEqual(raw_fa, 2.0)

        # Under 3-of-5 persistence rule, sporadic false alerts are filtered out -> 0 alerts / hour
        persisted_fa = compute_false_alerts_per_hour(y_true, y_pred, window_step_sec=30.0, persistence_rule=True)
        self.assertAlmostEqual(persisted_fa, 0.0)

    def test_bootstrap_confidence_intervals(self):
        """Test 1,000 resample bootstrap CI generation over subject scores."""
        subject_scores = [{"macro_f1": 0.80}, {"macro_f1": 0.85}, {"macro_f1": 0.90}, {"macro_f1": 0.88}]
        mean_val, ci_low, ci_high = bootstrap_subject_confidence_intervals(subject_scores, metric_name="macro_f1")
        self.assertAlmostEqual(mean_val, 0.8575, delta=0.01)
        self.assertLessEqual(ci_low, mean_val)
        self.assertGreaterEqual(ci_high, mean_val)

    def test_calibration_and_selective_prediction(self):
        """Test probability calibrator and selective prediction rejection."""
        val_probs = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        y_val = np.array([0, 0, 0, 1, 1, 1])

        calibrator = SubjectCalibrator(method="sigmoid").fit(val_probs, y_val)
        calibrated = calibrator.predict_proba(val_probs)
        self.assertEqual(len(calibrated), len(val_probs))

        # Test selective prediction
        df_selective = evaluate_selective_prediction(y_val, calibrated, delta_steps=5)
        self.assertIn("coverage", df_selective.columns)
        self.assertIn("macro_f1", df_selective.columns)
        # As delta increases, coverage must decrease or remain equal
        self.assertGreaterEqual(df_selective["coverage"].iloc[0], df_selective["coverage"].iloc[-1])

    def test_loso_evaluation_pipeline_with_leakage_assertion(self):
        """Test LOSO evaluation on a multi-subject DataFrame enforcing zero subject leakage."""
        np.random.seed(42)
        n_samples_per_sub = 20
        subjects = ["S2", "S3", "S4"]
        records = []

        for sub in subjects:
            for i in range(n_samples_per_sub):
                label = 1 if i % 2 == 0 else 0
                records.append({
                    "dataset": "WESAD",
                    "subject_id": sub,
                    "feat_1": np.random.randn() + (1.5 if label == 1 else -1.5),
                    "feat_2": np.random.randn() + (1.0 if label == 1 else -1.0),
                    "label": label,
                })
        df_toy = pd.DataFrame(records)

        res_df, summary = run_loso_evaluation(
            df_toy,
            model_name="m0_logistic_regression",
            feature_cols=["feat_1", "feat_2"],
            dataset_name="WESAD",
        )
        self.assertEqual(len(res_df), 3) # Evaluated on each of the 3 subjects
        self.assertIn("macro_f1_mean", summary)
        self.assertIn("macro_f1_95ci_lower", summary)

    def test_cnn_loso_evaluation_pipeline(self):
        """Test 1D-CNN M2 LOSO evaluation with strict subject isolation."""
        np.random.seed(42)
        subjects = np.array(["S2"] * 6 + ["S3"] * 6 + ["S4"] * 6)
        labels = np.array([0, 0, 0, 1, 1, 1] * 3, dtype=np.float32)
        tensors = np.random.randn(18, 6, 128).astype(np.float32)
        tensors[labels == 1] += 1.2

        res_df, summary = run_cnn_loso_evaluation(
            tensors=tensors,
            labels=labels,
            subject_ids=subjects,
            dataset_name="WESAD",
            epochs=3,
            batch_size=6,
            patience=2,
        )
        self.assertEqual(len(res_df), 3)
        self.assertEqual(summary["model"], "m2_cnn_1d")
        self.assertIn("macro_f1_mean", summary)


if __name__ == "__main__":
    unittest.main()
