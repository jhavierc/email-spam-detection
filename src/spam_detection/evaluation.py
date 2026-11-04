"""Evaluation helpers for the classifier."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, average_precision_score)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": threshold,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "classification_report": classification_report(
            y_true, y_pred, target_names=["ham", "spam"], output_dict=True, zero_division=0),
    }


def best_threshold(y_true, y_prob, grid=None) -> tuple[float, float]:
    """Return (threshold, f1) maximising F1 for the spam class."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    scores = [(t, f1_score(y_true, (np.asarray(y_prob) >= t).astype(int), zero_division=0)) for t in grid]
    t, f = max(scores, key=lambda x: x[1])
    return float(t), float(f)
