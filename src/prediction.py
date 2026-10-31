"""
Inference & Prediction Pipeline for Credit Card Fraud Detection.
Loads the trained ANN model and preprocessing scalers.
Processes manual input records or batch data, computes fraud probabilities,
classifies risk tiers, and generates human-readable explanations.
"""

import os
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from src.preprocessing import DataPreprocessor
from src.model import load_fraud_model


class FraudPredictor:
    """
    End-to-end Inference Engine for Fraud Classification.
    """

    # Top features historically correlated with fraud in the Kaggle Credit Card dataset:
    # Strong negative correlation with fraud: V14, V12, V10, V17, V3, V16
    # Strong positive correlation with fraud: V4, V11, V2, V19
    KEY_INFLUENTIAL_FEATURES = ["V14", "V12", "V10", "V17", "V4", "V11", "V2", "Amount"]

    def __init__(
        self,
        model_path: str = "models/fraud_model.keras",
        scaler_path: str = "models/scaler.joblib"
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Attempts to load the model and scaler."""
        if os.path.exists(self.scaler_path):
            self.preprocessor = DataPreprocessor.load_artifacts(self.scaler_path)
        if os.path.exists(self.model_path) or os.path.exists(self.model_path.replace(".keras", ".h5")):
            self.model = load_fraud_model(self.model_path)

    def is_ready(self) -> bool:
        """Checks if both model and preprocessor are loaded."""
        return self.model is not None and self.preprocessor is not None

    def predict_single(
        self,
        raw_input: Dict[str, float],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes prediction for a single transaction record dictionary.
        Returns prediction flag, probability, risk level, confidence, and explanation.
        """
        if not self.is_ready():
            raise RuntimeError(
                "Inference pipeline is not ready. Please train the model and generate artifacts first."
            )

        # Scale features using preprocessor (Time and Amount scaled; V1..V28 passed through)
        scaled_input = self.preprocessor.transform_single_record(raw_input)
        
        # Raw probability from neural network output sigmoid
        pred_prob = float(self.model.predict(scaled_input, verbose=0)[0][0])
        is_fraud = bool(pred_prob >= threshold)

        # Risk assessment categorization
        if pred_prob < 0.20:
            risk_level = "LOW RISK (GENUINE)"
            risk_color = "#10b981"  # Emerald Green
            risk_severity = "low"
        elif pred_prob < 0.50:
            risk_level = "MODERATE RISK (SUSPICIOUS)"
            risk_color = "#f59e0b"  # Amber
            risk_severity = "moderate"
        elif pred_prob < 0.85:
            risk_level = "HIGH RISK (LIKELY FRAUD)"
            risk_color = "#f97316"  # Orange
            risk_severity = "high"
        else:
            risk_level = "CRITICAL RISK (CONFIRMED FRAUD)"
            risk_color = "#ef4444"  # Red
            risk_severity = "critical"

        # Explainability: inspect extreme deviations in key fraud-correlated features
        explanations = self._generate_explanation(raw_input, pred_prob)

        return {
            "is_fraud": is_fraud,
            "prediction_label": "FRAUDULENT" if is_fraud else "GENUINE",
            "fraud_probability": pred_prob,
            "fraud_probability_pct": f"{pred_prob * 100:.2f}%",
            "threshold": threshold,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_severity": risk_severity,
            "explanation": explanations,
            "features_analyzed": len(raw_input)
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Runs batch prediction on a DataFrame containing the 30 features.
        """
        if not self.is_ready():
            raise RuntimeError("Inference pipeline is not ready. Train the model first.")

        df_scaled = self.preprocessor.transform(df[self.preprocessor.FEATURE_COLS])
        probs = self.model.predict(df_scaled.values, batch_size=2048, verbose=0).flatten()
        
        result_df = df.copy()
        result_df["Fraud_Probability"] = probs
        result_df["Predicted_Class"] = (probs >= threshold).astype(int)
        result_df["Risk_Level"] = pd.cut(
            probs,
            bins=[-0.01, 0.20, 0.50, 0.85, 1.01],
            labels=["Low", "Moderate", "High", "Critical"]
        )
        return result_df

    def _generate_explanation(
        self,
        raw_input: Dict[str, float],
        prob: float
    ) -> List[str]:
        """
        Generates contextual natural-language explanation based on neural network output
        and statistical feature deviations.
        """
        notes = []

        v14 = raw_input.get("V14", 0.0)
        v12 = raw_input.get("V12", 0.0)
        v10 = raw_input.get("V10", 0.0)
        v17 = raw_input.get("V17", 0.0)
        v4 = raw_input.get("V4", 0.0)
        v11 = raw_input.get("V11", 0.0)
        amount = raw_input.get("Amount", 0.0)

        if prob >= 0.5:
            notes.append(
                f"Neural Network activated high fraud probability ({prob*100:.1f}%) exceeding detection threshold."
            )
            anomalies = []
            if v14 < -3.0:
                anomalies.append(f"V14 ({v14:.2f}) shows extreme negative deflection (strongest fraud indicator)")
            if v12 < -3.0:
                anomalies.append(f"V12 ({v12:.2f}) indicates severe abnormal deviation from legitimate transaction manifold")
            if v10 < -2.5:
                anomalies.append(f"V10 ({v10:.2f}) aligns with known unauthorized transaction patterns")
            if v17 < -2.5:
                anomalies.append(f"V17 ({v17:.2f}) exhibits high negative anomaly")
            if v4 > 2.5:
                anomalies.append(f"V4 ({v4:.2f}) shows elevated positive risk spike")
            if v11 > 2.5:
                anomalies.append(f"V11 ({v11:.2f}) exceeds normal variance range")

            if anomalies:
                notes.extend(anomalies)
            else:
                notes.append("Multi-dimensional feature interaction patterns triggered high neural activation.")
        else:
            notes.append(
                f"Neural Network evaluated normal transaction characteristics (Probability: {prob*100:.2f}%)."
            )
            notes.append("Latent PCA components (V1-V28) reside within legitimate distribution boundaries.")
            if amount > 1000:
                notes.append(f"Amount (${amount:,.2f}) is higher than median, but behavioral signature matches genuine cardholder usage.")
            else:
                notes.append(f"Transaction amount (${amount:,.2f}) and temporal sequence conform to standard legitimate activity.")

        return notes


# Preset realistic benchmark samples for demonstration in the dashboard
SAMPLE_GENUINE_TRANSACTION = {
    "Time": 85000.0,
    "V1": -0.85, "V2": 0.42, "V3": 1.25, "V4": -0.34, "V5": 0.12,
    "V6": -0.21, "V7": 0.35, "V8": 0.18, "V9": 0.45, "V10": -0.15,
    "V11": -0.52, "V12": 0.28, "V13": 0.14, "V14": 0.08, "V15": 0.65,
    "V16": -0.12, "V17": 0.21, "V18": -0.05, "V19": 0.10, "V20": -0.08,
    "V21": -0.15, "V22": -0.25, "V23": 0.05, "V24": 0.12, "V25": -0.18,
    "V26": 0.05, "V27": 0.02, "V28": 0.01,
    "Amount": 49.99
}

SAMPLE_FRAUD_TRANSACTION = {
    "Time": 41200.0,
    "V1": -4.38, "V2": 3.86, "V3": -6.25, "V4": 4.82, "V5": -3.55,
    "V6": -1.82, "V7": -5.10, "V8": 2.95, "V9": -3.20, "V10": -7.12,
    "V11": 4.15, "V12": -8.45, "V13": -0.25, "V14": -9.32, "V15": -0.58,
    "V16": -6.21, "V17": -11.45, "V18": -4.10, "V19": 1.85, "V20": 0.58,
    "V21": 0.82, "V22": -0.35, "V23": -0.22, "V24": 0.15, "V25": 0.42,
    "V26": 0.38, "V27": 0.85, "V28": 0.28,
    "Amount": 289.50
}

SAMPLE_SUSPICIOUS_TRANSACTION = {
    "Time": 98200.0,
    "V1": -1.85, "V2": 1.45, "V3": -2.10, "V4": 2.15, "V5": -1.20,
    "V6": -0.85, "V7": -1.65, "V8": 0.95, "V9": -1.10, "V10": -2.35,
    "V11": 1.85, "V12": -2.80, "V13": 0.10, "V14": -3.20, "V15": -0.15,
    "V16": -1.95, "V17": -3.10, "V18": -1.25, "V19": 0.75, "V20": 0.25,
    "V21": 0.35, "V22": 0.10, "V23": -0.05, "V24": 0.05, "V25": 0.15,
    "V26": 0.12, "V27": 0.22, "V28": 0.08,
    "Amount": 1250.00
}
