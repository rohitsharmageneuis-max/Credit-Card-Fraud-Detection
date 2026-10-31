"""
Data Preprocessing Module for Credit Card Fraud Detection.
Handles dataset loading, exploratory data checks, stratified splitting,
leakage-free feature scaling (Time & Amount), and SMOTE oversampling on training data only.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE


class DataPreprocessor:
    """
    Robust Preprocessing Pipeline for Credit Card Fraud Detection.
    Prevents data leakage by fitting scalers strictly on training data.
    Applies SMOTE oversampling exclusively on training split.
    """

    FEATURE_COLS = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]
    TARGET_COL = 'Class'

    def __init__(self, scaler_type: str = "robust"):
        """
        Initialize preprocessor.
        :param scaler_type: 'robust' (RobustScaler, less sensitive to outliers) or 'standard'
        """
        self.scaler_type = scaler_type
        if scaler_type == "robust":
            self.amount_scaler = RobustScaler()
            self.time_scaler = RobustScaler()
        else:
            self.amount_scaler = StandardScaler()
            self.time_scaler = StandardScaler()
            
        self.is_fitted = False
        self.feature_columns = self.FEATURE_COLS.copy()

    @staticmethod
    def load_data(filepath: str = "data/creditcard.csv") -> pd.DataFrame:
        """
        Loads CSV dataset and verifies columns.
        Raises FileNotFoundError with detailed instructions if file is missing.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"[DATASET MISSING] Could not find dataset at '{filepath}'.\n"
                f"Please download the Kaggle Credit Card Fraud dataset from:\n"
                f"https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                f"and place 'creditcard.csv' into the '{os.path.dirname(filepath)}/' directory."
            )
        df = pd.read_csv(filepath)
        return df

    @staticmethod
    def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts structural, missing value, duplicate, and class balance statistics.
        """
        missing_counts = df.isnull().sum()
        total_missing = int(missing_counts.sum())
        duplicate_count = int(df.duplicated().sum())
        
        class_counts = df['Class'].value_counts().to_dict()
        genuine_count = int(class_counts.get(0, 0))
        fraud_count = int(class_counts.get(1, 0))
        total_tx = len(df)
        fraud_pct = (fraud_count / total_tx * 100) if total_tx > 0 else 0.0

        summary = {
            "total_rows": total_tx,
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": total_missing,
            "missing_per_column": missing_counts[missing_counts > 0].to_dict(),
            "duplicate_rows": duplicate_count,
            "genuine_count": genuine_count,
            "fraud_count": fraud_count,
            "fraud_percentage": fraud_pct,
            "amount_stats": {
                "mean": float(df['Amount'].mean()),
                "median": float(df['Amount'].median()),
                "min": float(df['Amount'].min()),
                "max": float(df['Amount'].max()),
                "std": float(df['Amount'].std()),
            }
        }
        return summary

    def fit_transform_train(
        self, X_train: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fits scalers ONLY on X_train for Amount and Time.
        Returns scaled X_train DataFrame.
        """
        X_train_scaled = X_train.copy()
        
        # Fit and transform Amount
        X_train_scaled['Amount'] = self.amount_scaler.fit_transform(
            X_train[['Amount']]
        )
        
        # Fit and transform Time
        X_train_scaled['Time'] = self.time_scaler.fit_transform(
            X_train[['Time']]
        )
        
        self.is_fitted = True
        return X_train_scaled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms test or new data using ALREADY FITTED scalers.
        Prevents data leakage.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor scalers have not been fitted yet! Call fit_transform_train first.")
        
        X_scaled = X.copy()
        X_scaled['Amount'] = self.amount_scaler.transform(X[['Amount']])
        X_scaled['Time'] = self.time_scaler.transform(X[['Time']])
        return X_scaled

    def apply_smote(
        self,
        X_train_scaled: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies SMOTE oversampling strictly to the training split.
        """
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
        return X_res, y_res

    def run_pipeline(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        apply_smote: bool = True,
        drop_duplicates: bool = False,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        End-to-end preprocessing pipeline:
        1. Optional duplicate removal
        2. Feature/Target split
        3. Stratified Train-Test Split
        4. Fit & Transform Scaler on Train; Transform Test (no leakage)
        5. SMOTE oversampling on Train only
        """
        data = df.copy()
        if drop_duplicates:
            data = data.drop_duplicates()

        X = data[self.FEATURE_COLS]
        y = data[self.TARGET_COL]

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Scale features without data leakage
        X_train_scaled = self.fit_transform_train(X_train)
        X_test_scaled = self.transform(X_test)

        # Apply SMOTE to training data only
        if apply_smote:
            X_train_final, y_train_final = self.apply_smote(
                X_train_scaled, y_train, random_state=random_state
            )
        else:
            X_train_final = X_train_scaled.values
            y_train_final = y_train.values

        X_test_final = X_test_scaled.values
        y_test_final = y_test.values

        meta = {
            "train_samples_before_smote": len(X_train),
            "train_samples_after_smote": len(X_train_final),
            "test_samples": len(X_test_final),
            "train_fraud_count_before": int(y_train.sum()),
            "train_genuine_count_before": int((y_train == 0).sum()),
            "train_fraud_count_after": int((y_train_final == 1).sum()),
            "train_genuine_count_after": int((y_train_final == 0).sum()),
            "test_fraud_count": int(y_test.sum()),
            "test_genuine_count": int((y_test == 0).sum()),
        }

        return X_train_final, y_train_final, X_test_final, y_test_final, meta

    def save_artifacts(self, save_dir: str = "models") -> str:
        """
        Saves the fitted scaler objects and metadata using joblib.
        """
        os.makedirs(save_dir, exist_ok=True)
        scaler_payload = {
            "amount_scaler": self.amount_scaler,
            "time_scaler": self.time_scaler,
            "scaler_type": self.scaler_type,
            "feature_cols": self.FEATURE_COLS,
            "is_fitted": self.is_fitted
        }
        path = os.path.join(save_dir, "scaler.joblib")
        joblib.dump(scaler_payload, path)
        return path

    @classmethod
    def load_artifacts(cls, filepath: str = "models/scaler.joblib") -> "DataPreprocessor":
        """
        Loads fitted scaler objects from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler artifact not found at {filepath}")
        payload = joblib.load(filepath)
        instance = cls(scaler_type=payload.get("scaler_type", "robust"))
        instance.amount_scaler = payload["amount_scaler"]
        instance.time_scaler = payload["time_scaler"]
        instance.is_fitted = payload.get("is_fitted", True)
        instance.feature_columns = payload.get("feature_cols", cls.FEATURE_COLS)
        return instance

    def transform_single_record(self, record_dict: Dict[str, float]) -> np.ndarray:
        """
        Takes raw dictionary with all 30 features (Time, V1..V28, Amount),
        scales Time and Amount with fitted scalers, and returns a 2D numpy array (1, 30).
        """
        df_single = pd.DataFrame([record_dict])
        missing_cols = [c for c in self.FEATURE_COLS if c not in df_single.columns]
        if missing_cols:
            raise ValueError(f"Missing required input features: {missing_cols}")
        
        # Ensure correct column ordering
        df_single = df_single[self.FEATURE_COLS]
        scaled = self.transform(df_single)
        return scaled.values.astype(np.float32)
