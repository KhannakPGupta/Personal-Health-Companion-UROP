"""Unit tests for Phase 3 Federated Edge Simulation (FedAvg & FedProx)."""

import unittest
import numpy as np
import torch

from src.federated.fedavg import (
    EdgeClientData,
    federated_weighted_average,
    compute_communication_cost_kb,
    run_federated_training,
)
from src.federated.federated_evaluator import run_federated_loso_evaluation
from src.eval.run_phase1_experiments import generate_synthetic_cohort


class TestPhase3Federated(unittest.TestCase):
    def test_federated_weighted_average_exactness(self):
        """Test that sample-proportional FedAvg aggregation computes exact weighted mean."""
        s1 = {"w": torch.tensor([1.0, 2.0]), "steps": torch.tensor(5, dtype=torch.long)}
        s2 = {"w": torch.tensor([4.0, 8.0]), "steps": torch.tensor(5, dtype=torch.long)}

        # Client 1 has 10 samples (weight 0.25), Client 2 has 30 samples (weight 0.75)
        agg = federated_weighted_average([s1, s2], [10, 30])
        expected_w = torch.tensor([0.25 * 1.0 + 0.75 * 4.0, 0.25 * 2.0 + 0.75 * 8.0])
        self.assertTrue(torch.allclose(agg["w"], expected_w))
        self.assertEqual(int(agg["steps"].item()), 5)

    def test_communication_cost_accounting(self):
        """Test float32 vs int8 communication overhead calculation."""
        f32 = compute_communication_cost_kb(num_parameters=10240, num_rounds=5, clients_per_round=4, quantized_int8=False)
        int8 = compute_communication_cost_kb(num_parameters=10240, num_rounds=5, clients_per_round=4, quantized_int8=True)
        # 10240 params * 4 bytes = 40 KB per payload; int8 = 10 KB per payload (4x reduction)
        self.assertAlmostEqual(f32["single_model_payload_kb"], 40.0, places=2)
        self.assertAlmostEqual(int8["single_model_payload_kb"], 10.0, places=2)
        self.assertAlmostEqual(f32["total_communication_kb"] / int8["total_communication_kb"], 4.0, places=2)

    def test_federated_training_and_leakage_rejection(self):
        """Test FedAvg and FedProx training and assert subject leakage raises AssertionError."""
        c1 = EdgeClientData("S2", torch.randn(8, 6, 64), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32))
        c2 = EdgeClientData("S3", torch.randn(8, 6, 64), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32))

        # Test leakage rejection if test_subjects contains S2
        with self.assertRaises(AssertionError):
            run_federated_training([c1, c2], test_subjects=["S2"], rounds=2, local_epochs=1)

        # Valid disjoint test subject S4
        model, logs = run_federated_training(
            [c1, c2], test_subjects=["S4"], rounds=2, local_epochs=1, fedprox_mu=0.05
        )
        self.assertEqual(len(logs), 2)
        self.assertGreater(model.count_parameters(), 0)
        self.assertLess(model.count_parameters(), 50000)

    def test_federated_loso_evaluator(self):
        """Test end-to-end Federated LOSO evaluator across multiple subjects."""
        df, tensors = generate_synthetic_cohort(
            "WESAD", ["S2", "S3", "S4"], windows_per_subject=8, seq_len=64, seed=42
        )
        res_df, summary = run_federated_loso_evaluation(
            tensors=tensors,
            labels=df["label"].values,
            subject_ids=df["subject_id"].values,
            dataset_name="WESAD",
            algorithm_name="FedAvg",
            rounds=2,
            local_epochs=1,
        )
        self.assertEqual(len(res_df), 3)
        self.assertIn("macro_f1_mean", summary)
        self.assertIn("total_comm_int8_kb", summary)


if __name__ == "__main__":
    unittest.main()

