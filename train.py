"""
End-to-End Training Script for AI Credit Card Fraud Detection System.
Executes data loading, inspection, stratified splitting, leakage-free scaling,
SMOTE oversampling on training set only, ANN compilation, training with callbacks,
test set evaluation, and artifact persistence.

Usage:
    python train.py [--data_path data/creditcard.csv] [--epochs 25] [--batch_size 1024]
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import DataPreprocessor
from src.model import build_fraud_ann, get_training_callbacks, train_model
from src.evaluation import evaluate_model_predictions, save_metrics_and_history


def print_banner():
    banner = """
========================================================================
    AI CREDIT CARD FRAUD DETECTION - DEEP LEARNING (ANN) PIPELINE
========================================================================
    """
    print(banner)


def check_dataset_exists(data_path: str) -> bool:
    """Verifies that the dataset file exists before starting."""
    if not os.path.exists(data_path):
        print(f"\n[ERROR] Missing required dataset: '{data_path}'")
        print("-" * 72)
        print("INSTRUCTIONS TO DOWNLOAD THE DATASET:")
        print("1. Visit the official Kaggle dataset page:")
        print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. Download the 'archive.zip' or 'creditcardfraud.zip' archive.")
        print(f"3. Extract 'creditcard.csv' and place it into: '{os.path.abspath(data_path)}'")
        print("   Or if Kaggle CLI is installed:")
        print(f"   kaggle datasets download -d mlg-ulb/creditcardfraud -p {os.path.dirname(data_path)} --unzip")
        print("-" * 72)
        print("[POLICY] Synthesizing fake training data is disabled. Exiting.")
        return False
    return True


def run_training_pipeline(args):
    print_banner()

    data_path = args.data_path
    if not check_dataset_exists(data_path):
        sys.exit(1)

    print(f"\n[1/5] Loading Dataset from: {data_path}...")
    start_time = time.time()
    df = DataPreprocessor.load_data(data_path)
    print(f"-> Loaded {len(df):,} transactions in {time.time() - start_time:.2f}s")

    print("\n[2/5] Inspecting Dataset Structure & Imbalance...")
    summary = DataPreprocessor.inspect_data(df)
    print(f"-> Total Rows: {summary['total_rows']:,}")
    print(f"-> Total Columns: {summary['total_columns']}")
    print(f"-> Missing Values: {summary['missing_values']}")
    print(f"-> Duplicate Rows: {summary['duplicate_rows']:,}")
    print(f"-> Genuine Transactions (Class 0): {summary['genuine_count']:,} ({100 - summary['fraud_percentage']:.3f}%)")
    print(f"-> Fraudulent Transactions (Class 1): {summary['fraud_count']:,} ({summary['fraud_percentage']:.4f}%)")
    print(f"-> Imbalance Ratio: ~1 fraud per {summary['genuine_count'] // max(1, summary['fraud_count'])} genuine transactions")

    print(f"\n[3/5] Preprocessing (Stratified Split, Scaling, SMOTE)...")
    preprocessor = DataPreprocessor(scaler_type=args.scaler)
    X_train_res, y_train_res, X_test_scaled, y_test, meta = preprocessor.run_pipeline(
        df=df,
        test_size=args.test_size,
        apply_smote=not args.no_smote,
        drop_duplicates=args.drop_duplicates,
        random_state=args.seed
    )

    print(f"-> Training Set (before SMOTE): {meta['train_samples_before_smote']:,} samples "
          f"({meta['train_fraud_count_before']:,} fraud)")
    if not args.no_smote:
        print(f"-> Training Set (after SMOTE): {meta['train_samples_after_smote']:,} balanced samples "
              f"({meta['train_fraud_count_after']:,} fraud / {meta['train_genuine_count_after']:,} genuine)")
    print(f"-> Test Set (Untouched ground truth): {meta['test_samples']:,} samples ({meta['test_fraud_count']} fraud)")

    # Save fitted scalers
    os.makedirs(args.model_dir, exist_ok=True)
    scaler_saved_path = preprocessor.save_artifacts(args.model_dir)
    print(f"-> Preprocessing scalers saved to: {scaler_saved_path}")

    print(f"\n[4/5] Building & Compiling Artificial Neural Network (ANN)...")
    input_dim = X_train_res.shape[1]
    model = build_fraud_ann(input_dim=input_dim, learning_rate=args.learning_rate)
    model.summary()

    model_checkpoint_path = os.path.join(args.model_dir, "fraud_model.keras")
    callbacks_list = get_training_callbacks(checkpoint_filepath=model_checkpoint_path, patience=args.patience)

    print(f"\n[5/5] Training ANN Model (Epochs: {args.epochs}, Batch Size: {args.batch_size})...")
    history = train_model(
        model=model,
        X_train=X_train_res,
        y_train=y_train_res,
        X_val=X_test_scaled,
        y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_filepath=model_checkpoint_path,
        callbacks_list=callbacks_list
    )

    print("\n" + "=" * 72)
    print("               MODEL EVALUATION ON TEST SET (NO LEAKAGE)")
    print("=" * 72)
    test_probs = model.predict(X_test_scaled, batch_size=args.batch_size, verbose=0).flatten()
    metrics_result = evaluate_model_predictions(y_test, test_probs, threshold=args.threshold)

    print(f"Accuracy:        {metrics_result['accuracy']:.4f}")
    print(f"Precision:       {metrics_result['precision']:.4f}")
    print(f"Recall:          {metrics_result['recall']:.4f}")
    print(f"F1-Score:        {metrics_result['f1_score']:.4f}")
    print(f"ROC-AUC:         {metrics_result['roc_auc']:.4f}")
    print(f"PR-AUC:          {metrics_result['pr_auc']:.4f}")
    print("\nConfusion Matrix:")
    cm = np.array(metrics_result['confusion_matrix'])
    print(f"  [Legitimate] TN: {cm[0, 0]:,}  |  FP: {cm[0, 1]:,}")
    print(f"  [Fraudulent] FN: {cm[1, 0]:,}  |  TP: {cm[1, 1]:,}")

    metrics_file, history_file = save_metrics_and_history(
        metrics_dict=metrics_result,
        history_dict=history.history,
        save_dir=args.model_dir
    )
    print(f"\nArtifacts successfully persisted:")
    print(f"-> ANN Model:       {model_checkpoint_path}")
    print(f"-> Scaler Pipeline: {scaler_saved_path}")
    print(f"-> Test Metrics:    {metrics_file}")
    print(f"-> Training History:{history_file}")
    print("=" * 72)
    print("[SUCCESS] Training pipeline completed successfully!\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Credit Card Fraud Detection ANN")
    parser.add_argument("--data_path", type=str, default="data/creditcard.csv", help="Path to creditcard.csv")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save artifacts")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Stratified test split size")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--scaler", type=str, default="robust", choices=["robust", "standard"], help="Scaler type")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold")
    parser.add_argument("--no_smote", action="store_true", help="Disable SMOTE oversampling")
    parser.add_argument("--drop_duplicates", action="store_true", help="Drop duplicate transactions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training_pipeline(args)
