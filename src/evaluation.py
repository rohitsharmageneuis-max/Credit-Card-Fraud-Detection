"""
Model Evaluation and Interactive Visualization Module.
Calculates rigorous classification metrics:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix.
Builds Plotly charts matching the dark fintech theme.
Saves and loads genuine evaluation metrics and training history.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)
import plotly.graph_objects as go
import plotly.express as px


# Color palette for Dark Fintech Theme
PALETTE = {
    "bg_paper": "#111827",
    "bg_plot": "#0b0f19",
    "text": "#f8fafc",
    "text_muted": "#94a3b8",
    "grid": "rgba(148, 163, 184, 0.12)",
    "cyan": "#06b6d4",
    "cyan_light": "#22d3ee",
    "purple": "#8b5cf6",
    "purple_light": "#a855f7",
    "red": "#ef4444",
    "green": "#10b981",
    "gold": "#f59e0b"
}


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes all standard binary classification metrics from ground truth and predicted probabilities.
    No fabricated metrics are generated.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_pred_probs))
    pr_auc = float(average_precision_score(y_true, y_pred_probs))
    cm = confusion_matrix(y_true, y_pred).tolist()

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_probs)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_probs)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    results = {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "curves": {
            "fpr": [float(x) for x in fpr[::max(1, len(fpr) // 200)]],
            "tpr": [float(x) for x in tpr[::max(1, len(tpr) // 200)]],
            "precision_curve": [float(x) for x in precision_curve[::max(1, len(precision_curve) // 200)]],
            "recall_curve": [float(x) for x in recall_curve[::max(1, len(recall_curve) // 200)]]
        }
    }
    return results


def save_metrics_and_history(
    metrics_dict: Dict[str, Any],
    history_dict: Optional[Dict[str, Any]] = None,
    save_dir: str = "models"
) -> Tuple[str, Optional[str]]:
    """
    Saves true calculated evaluation metrics and training history to JSON files.
    """
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
    history_path = None
    if history_dict is not None:
        history_path = os.path.join(save_dir, "history.json")
        # Ensure values are JSON serializable
        clean_hist = {
            k: [float(v) for v in vals] if isinstance(vals, (list, np.ndarray)) else vals
            for k, vals in history_dict.items()
        }
        with open(history_path, "w") as f:
            json.dump(clean_hist, f, indent=4)

    return metrics_path, history_path


def load_metrics_and_history(save_dir: str = "models") -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Loads saved evaluation metrics and history JSON if available.
    Returns (None, None) if not found.
    """
    metrics_path = os.path.join(save_dir, "metrics.json")
    history_path = os.path.join(save_dir, "history.json")

    metrics_data = None
    history_data = None

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history_data = json.load(f)

    return metrics_data, history_data


# =====================================================================
# PLOTLY CHART BUILDERS FOR FINTECH DASHBOARD
# =====================================================================

def plot_confusion_matrix(cm: list) -> go.Figure:
    """
    Creates an interactive Confusion Matrix heatmap styled for the dark fintech theme.
    """
    cm_arr = np.array(cm)
    labels = ["Genuine (0)", "Fraud (1)"]
    
    # Text annotation with counts and percentages
    total = cm_arr.sum()
    text_annot = [
        [f"{val:,}<br>({val/total:.2%})" for val in row]
        for row in cm_arr
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=labels,
        y=labels,
        text=text_annot,
        texttemplate="%{text}",
        textfont={"size": 15, "color": "#ffffff", "family": "Plus Jakarta Sans"},
        colorscale=[
            [0.0, "#0f172a"],
            [0.5, "#4338ca"],
            [1.0, "#06b6d4"]
        ],
        colorbar=dict(title=dict(text="Count", font=dict(color=PALETTE["text"])), tickfont=dict(color=PALETTE["text"])),
        showscale=True
    ))

    fig.update_layout(
        title=dict(text="<b>Confusion Matrix</b>", font=dict(color=PALETTE["text"], size=16)),
        xaxis=dict(title=dict(text="Predicted Class", font=dict(color=PALETTE["text_muted"])), tickfont=dict(color=PALETTE["text"])),
        yaxis=dict(title=dict(text="Actual Class", font=dict(color=PALETTE["text_muted"])), tickfont=dict(color=PALETTE["text"]), autorange="reversed"),
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig


def plot_roc_curve(fpr: list, tpr: list, roc_auc: float) -> go.Figure:
    """
    Creates an interactive ROC curve chart.
    """
    fig = go.Figure()

    # ROC Line
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ANN ROC (AUC = {roc_auc:.4f})",
        line=dict(color=PALETTE["cyan"], width=3),
        fill="tozeroy",
        fillcolor="rgba(6, 182, 212, 0.12)"
    ))

    # Diagonal Random Chance
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Classifier (AUC = 0.50)",
        line=dict(color=PALETTE["text_muted"], width=2, dash="dash")
    ))

    fig.update_layout(
        title=dict(text=f"<b>Receiver Operating Characteristic (ROC)</b> — AUC: {roc_auc:.4f}", font=dict(color=PALETTE["text"], size=16)),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"], range=[0, 1]),
        yaxis=dict(title="True Positive Rate (Recall)", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"], range=[0, 1.02]),
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        legend=dict(font=dict(color=PALETTE["text"]), bgcolor="rgba(17, 24, 39, 0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig


def plot_precision_recall_curve(recall_curve: list, precision_curve: list, pr_auc: float) -> go.Figure:
    """
    Creates an interactive Precision-Recall curve chart (crucial for imbalanced fraud datasets).
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall_curve, y=precision_curve,
        mode="lines",
        name=f"ANN PR (PR-AUC = {pr_auc:.4f})",
        line=dict(color=PALETTE["purple"], width=3),
        fill="tozeroy",
        fillcolor="rgba(139, 92, 246, 0.12)"
    ))

    fig.update_layout(
        title=dict(text=f"<b>Precision-Recall Curve</b> — PR-AUC: {pr_auc:.4f}", font=dict(color=PALETTE["text"], size=16)),
        xaxis=dict(title="Recall (Sensitivity)", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"], range=[0, 1.02]),
        yaxis=dict(title="Precision", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"], range=[0, 1.02]),
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        legend=dict(font=dict(color=PALETTE["text"]), bgcolor="rgba(17, 24, 39, 0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig


def plot_training_history(history: Dict[str, list]) -> Tuple[go.Figure, go.Figure]:
    """
    Builds two figures:
    1. Training & Validation Loss
    2. Training & Validation Accuracy / AUC
    """
    epochs = list(range(1, len(history.get("loss", [])) + 1))

    # Loss Figure
    fig_loss = go.Figure()
    if "loss" in history:
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history["loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color=PALETTE["cyan"], width=2)
        ))
    if "val_loss" in history:
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history["val_loss"],
            mode="lines+markers",
            name="Val Loss",
            line=dict(color=PALETTE["purple"], width=2, dash="dash")
        ))
    fig_loss.update_layout(
        title=dict(text="<b>Training & Validation Loss</b>", font=dict(color=PALETTE["text"], size=15)),
        xaxis=dict(title="Epoch", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"]),
        yaxis=dict(title="Loss (Binary Crossentropy)", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"]),
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        legend=dict(font=dict(color=PALETTE["text"]), bgcolor="rgba(17, 24, 39, 0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=340
    )

    # Accuracy / AUC Figure
    fig_metric = go.Figure()
    acc_key = "accuracy" if "accuracy" in history else ("acc" if "acc" in history else None)
    val_acc_key = "val_accuracy" if "val_accuracy" in history else ("val_acc" if "val_acc" in history else None)

    if acc_key and acc_key in history:
        fig_metric.add_trace(go.Scatter(
            x=epochs, y=history[acc_key],
            mode="lines+markers",
            name="Train Accuracy",
            line=dict(color=PALETTE["green"], width=2)
        ))
    if val_acc_key and val_acc_key in history:
        fig_metric.add_trace(go.Scatter(
            x=epochs, y=history[val_acc_key],
            mode="lines+markers",
            name="Val Accuracy",
            line=dict(color=PALETTE["gold"], width=2, dash="dash")
        ))
    if "auc" in history:
        fig_metric.add_trace(go.Scatter(
            x=epochs, y=history["auc"],
            mode="lines",
            name="Train AUC",
            line=dict(color=PALETTE["cyan_light"], width=1.5, dash="dot")
        ))
    if "val_auc" in history:
        fig_metric.add_trace(go.Scatter(
            x=epochs, y=history["val_auc"],
            mode="lines",
            name="Val AUC",
            line=dict(color=PALETTE["purple_light"], width=1.5, dash="dot")
        ))

    fig_metric.update_layout(
        title=dict(text="<b>Training & Validation Accuracy / AUC</b>", font=dict(color=PALETTE["text"], size=15)),
        xaxis=dict(title="Epoch", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"]),
        yaxis=dict(title="Metric Score", color=PALETTE["text_muted"], gridcolor=PALETTE["grid"]),
        paper_bgcolor=PALETTE["bg_paper"],
        plot_bgcolor=PALETTE["bg_plot"],
        legend=dict(font=dict(color=PALETTE["text"]), bgcolor="rgba(17, 24, 39, 0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=340
    )

    return fig_loss, fig_metric
