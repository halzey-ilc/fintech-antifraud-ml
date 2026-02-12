from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate_binary_classifier(*, y_true: Any, y_score: Any) -> dict[str, float]:
    """
    Evaluate using threshold-free metrics.
    - ROC-AUC
    - PR-AUC (Average Precision)

    y_true: array-like of {0,1}
    y_score: array-like of probabilities in [0,1]
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    roc = float(roc_auc_score(y_true_arr, y_score_arr))
    pr = float(average_precision_score(y_true_arr, y_score_arr))
    return {"roc_auc": roc, "pr_auc": pr}


def thresholds_table(*, y_true: Any, y_score: Any, thresholds: list[float]) -> list[dict[str, float]]:
    """
    Compute precision/recall/f1 for a list of thresholds.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    out: list[dict[str, float]] = []
    for t in thresholds:
        y_pred = (y_score_arr >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_arr, y_pred, average="binary", zero_division=0
        )
        out.append(
            {
                "threshold": float(t),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return out
