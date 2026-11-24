"""Evaluation metrics, threshold tuning and curve extraction."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    low: float = 0.05,
    high: float = 0.95,
    step: float = 0.01,
) -> tuple[float, float]:
    """Choose the probability threshold that maximizes F1 on a validation set."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(low, high + step / 2, step):
        score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = round(float(threshold), 4), float(score)
    return best_threshold, best_f1


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute classification and probability-quality metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "n_samples": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Extract ROC and precision-recall curve points for dashboards."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        "roc": {
            "fpr": [round(float(value), 6) for value in fpr],
            "tpr": [round(float(value), 6) for value in tpr],
        },
        "pr": {
            "precision": [round(float(value), 6) for value in precision],
            "recall": [round(float(value), 6) for value in recall],
        },
    }


def calibration_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Extract calibration curve points (observed vs predicted probability)."""
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return {
        "mean_predicted": [round(float(value), 6) for value in mean_predicted],
        "fraction_positive": [round(float(value), 6) for value in fraction_positive],
    }
