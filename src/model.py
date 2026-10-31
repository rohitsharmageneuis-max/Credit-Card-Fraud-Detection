"""
Deep Learning Model Architecture & Training Utilities for Fraud Detection.
Implements the exact Artificial Neural Network (ANN) specified with:
- 30 input features
- Dense(128) + BatchNormalization + Dropout(0.3)
- Dense(64) + BatchNormalization + Dropout(0.2)
- Dense(32)
- Output Dense(1, sigmoid)
Compiled with Adam, Binary Crossentropy, and Precision/Recall/AUC metrics.
"""

import os
from typing import List, Optional, Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, metrics


def build_fraud_ann(
    input_dim: int = 30,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Constructs and compiles the Deep Learning Artificial Neural Network (ANN)
    according to architectural specifications.
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,), name="input_features"),
        
        # Dense block 1
        layers.Dense(128, activation="relu", name="dense_128"),
        layers.BatchNormalization(name="batch_norm_1"),
        layers.Dropout(0.3, name="dropout_1"),
        
        # Dense block 2
        layers.Dense(64, activation="relu", name="dense_64"),
        layers.BatchNormalization(name="batch_norm_2"),
        layers.Dropout(0.2, name="dropout_2"),
        
        # Dense block 3
        layers.Dense(32, activation="relu", name="dense_32"),
        
        # Output layer (Binary classification)
        layers.Dense(1, activation="sigmoid", name="output_fraud_prob")
    ], name="Credit_Card_Fraud_ANN")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(curve="ROC", name="auc"),
            metrics.AUC(curve="PR", name="pr_auc")
        ]
    )
    return model


def get_training_callbacks(
    checkpoint_filepath: str = "models/fraud_model.keras",
    patience: int = 8
) -> List[tf.keras.callbacks.Callback]:
    """
    Sets up EarlyStopping, ModelCheckpoint, and dynamic Learning Rate Reduction.
    """
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )

    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    return [cb_early_stop, cb_checkpoint, cb_reduce_lr]


def train_model(
    model: tf.keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 25,
    batch_size: int = 1024,
    checkpoint_filepath: str = "models/fraud_model.keras",
    callbacks_list: Optional[List[tf.keras.callbacks.Callback]] = None
) -> tf.keras.callbacks.History:
    """
    Trains the ANN on the preprocessed training set with validation monitoring.
    """
    if callbacks_list is None:
        callbacks_list = get_training_callbacks(checkpoint_filepath)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    return history


def load_fraud_model(filepath: str = "models/fraud_model.keras") -> tf.keras.Model:
    """
    Loads saved Keras ANN model from disk. Supports both .keras and legacy .h5.
    """
    if not os.path.exists(filepath):
        # Fallback check for .h5 if .keras not found
        alt_path = filepath.replace(".keras", ".h5")
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at '{filepath}'. Please train the model first.")

    model = tf.keras.models.load_model(filepath)
    return model
