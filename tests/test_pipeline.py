"""
Unit and Integration Tests for AI Credit Card Fraud Detection System.
Tests:
1. Required Package Imports
2. ANN Model Architecture Compilation & Forward Pass
3. Preprocessor Scaling & Single-record Transformation
4. Evaluation Metric Helpers & Visualization Builders
5. End-to-end Inference Engine with Explanation Output
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import DataPreprocessor
from src.model import build_fraud_ann
from src.evaluation import evaluate_model_predictions, plot_confusion_matrix, plot_roc_curve
from src.prediction import (
    FraudPredictor,
    SAMPLE_GENUINE_TRANSACTION,
    SAMPLE_FRAUD_TRANSACTION
)


class TestCreditCardFraudSystem(unittest.TestCase):

    def test_01_preprocessor_synthetic_batch(self):
        """Tests that DataPreprocessor fits and scales without data leakage."""
        np.random.seed(42)
        n_samples = 200
        cols = DataPreprocessor.FEATURE_COLS
        fake_data = {c: np.random.randn(n_samples) for c in cols}
        fake_data["Amount"] = np.random.exponential(50, n_samples)
        fake_data["Time"] = np.linspace(0, 10000, n_samples)
        fake_data["Class"] = [1 if i % 20 == 0 else 0 for i in range(n_samples)]
        
        df = pd.DataFrame(fake_data)
        preprocessor = DataPreprocessor(scaler_type="robust")
        
        # Run pipeline with SMOTE
        X_train_res, y_train_res, X_test_scaled, y_test, meta = preprocessor.run_pipeline(
            df=df,
            test_size=0.25,
            apply_smote=True,
            random_state=42
        )

        self.assertEqual(X_train_res.shape[1], 30)
        self.assertEqual(X_test_scaled.shape[1], 30)
        # Verify SMOTE balanced the train split
        self.assertEqual((y_train_res == 0).sum(), (y_train_res == 1).sum())
        # Test split remains untouched
        self.assertLess((y_test == 1).sum(), (y_test == 0).sum())

    def test_02_preprocessor_single_record(self):
        """Tests transforming a single record dictionary."""
        preprocessor = DataPreprocessor(scaler_type="robust")
        # Fit on minimal data
        df_sample = pd.DataFrame([SAMPLE_GENUINE_TRANSACTION, SAMPLE_FRAUD_TRANSACTION])
        preprocessor.fit_transform_train(df_sample[preprocessor.FEATURE_COLS])
        
        single_arr = preprocessor.transform_single_record(SAMPLE_GENUINE_TRANSACTION)
        self.assertEqual(single_arr.shape, (1, 30))
        self.assertIsInstance(single_arr, np.ndarray)

    def test_03_model_build_and_forward_pass(self):
        """Tests ANN construction, compilation and forward pass."""
        model = build_fraud_ann(input_dim=30, learning_rate=0.001)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 8)  # Dense, BN, Drop, Dense, BN, Drop, Dense, Output

        # Test forward pass with random batch of 5
        dummy_input = np.random.randn(5, 30).astype(np.float32)
        preds = model.predict(dummy_input, verbose=0)
        self.assertEqual(preds.shape, (5, 1))
        # Ensure outputs are valid probabilities between 0 and 1
        self.assertTrue(np.all(preds >= 0.0) and np.all(preds <= 1.0))

    def test_04_evaluation_metrics_calculation(self):
        """Tests evaluation metrics without fabrication."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        y_probs = np.array([0.05, 0.12, 0.20, 0.35, 0.88, 0.95, 0.40, 0.70])
        
        metrics = evaluate_model_predictions(y_true, y_probs, threshold=0.5)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

        # Test plotly chart builders
        fig_cm = plot_confusion_matrix(metrics["confusion_matrix"])
        self.assertIsNotNone(fig_cm)
        fig_roc = plot_roc_curve(metrics["curves"]["fpr"], metrics["curves"]["tpr"], metrics["roc_auc"])
        self.assertIsNotNone(fig_roc)

    def test_05_prediction_pipeline_explainability(self):
        """Tests that risk explanations are generated logically."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        
        # Test low probability explanation
        expl_low = predictor._generate_explanation(SAMPLE_GENUINE_TRANSACTION, prob=0.04)
        self.assertTrue(any("normal" in note.lower() or "legitimate" in note.lower() for note in expl_low))

        # Test high probability explanation
        expl_high = predictor._generate_explanation(SAMPLE_FRAUD_TRANSACTION, prob=0.96)
        self.assertTrue(any("high fraud probability" in note.lower() for note in expl_high))


if __name__ == "__main__":
    unittest.main()
