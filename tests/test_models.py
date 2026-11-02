"""Unit tests for Phase 1D: Models, CNN parameter budget, and int8 quantization."""

import unittest
import numpy as np
import torch
from src.models.baselines import build_m0_models, build_m1_models
from src.models.cnn_1d import LightweightStrainCNN, train_cnn_with_early_stopping
from src.edge.quantization import quantize_model_int8, measure_model_size_kb, profile_quantization


class TestModelsAndEdge(unittest.TestCase):
    def test_m0_and_m1_builders(self):
        """Verify baseline model dictionaries instantiate correctly."""
        m0 = build_m0_models()
        self.assertIn("m0_majority", m0)
        self.assertIn("m0_logistic_regression", m0)

        m1 = build_m1_models()
        self.assertIn("m1_decision_tree_depth5", m1)
        self.assertIn("m1_random_forest", m1)
        self.assertIn("m1_hist_gradient_boosting", m1)

    def test_cnn_parameter_budget_under_50k(self):
        """Enforce strict edge constraint: M2 1D-CNN must have under 50k parameters."""
        model = LightweightStrainCNN(in_channels=6, num_classes=1)
        num_params = model.count_parameters()
        self.assertLess(num_params, 50000, f"Model exceeded 50k parameter budget! Found: {num_params}")
        # Expected to be ~13,000 parameters
        self.assertGreater(num_params, 1000)

    def test_cnn_forward_pass_shape(self):
        """Test CNN output shape on single window and batch input."""
        model = LightweightStrainCNN(in_channels=6, num_classes=1)
        model.eval()

        # Batch of 4 windows, 6 channels, 1920 samples
        batch_input = torch.randn(4, 6, 1920)
        with torch.no_grad():
            output = model(batch_input)
        self.assertEqual(output.shape, (4, 1))

        # Also supports transposed (B, 1920, 6)
        transposed_input = torch.randn(2, 1920, 6)
        with torch.no_grad():
            trans_output = model(transposed_input)
        self.assertEqual(trans_output.shape, (2, 1))

    def test_cnn_training_early_stopping(self):
        """Test training loop on small synthetic tensor batches."""
        model = LightweightStrainCNN(in_channels=6, num_classes=1)
        X_train = torch.randn(16, 6, 1920)
        y_train = torch.randint(0, 2, (16,)).float()
        X_val = torch.randn(8, 6, 1920)
        y_val = torch.randint(0, 2, (8,)).float()

        trained_model, history = train_cnn_with_early_stopping(
            model, X_train, y_train, X_val, y_val, epochs=2, batch_size=8, patience=2
        )
        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_loss"]), 2)

    def test_int8_quantization_and_profiling(self):
        """Test dynamic int8 quantization and size profiling."""
        model = LightweightStrainCNN(in_channels=6, num_classes=1)
        val_x = torch.randn(10, 6, 1920)
        val_y = torch.randint(0, 2, (10,)).float()

        quant_res = profile_quantization(model, val_x, val_y)
        self.assertIn("parameters", quant_res)
        self.assertIn("float_size_kb", quant_res)
        self.assertIn("int8_size_kb", quant_res)
        self.assertIn("delta_accuracy", quant_res)
        self.assertLessEqual(quant_res["parameters"], 50000)


if __name__ == "__main__":
    unittest.main()
