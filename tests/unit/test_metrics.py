from __future__ import annotations

import numpy as np

from app.evaluation.metrics import evaluate_binary_classifier, thresholds_table


def test_evaluate_binary_classifier_returns_expected_keys() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

    out = evaluate_binary_classifier(y_true=y_true, y_score=y_score)
    assert "roc_auc" in out
    assert "pr_auc" in out
    assert 0.0 <= out["roc_auc"] <= 1.0
    assert 0.0 <= out["pr_auc"] <= 1.0


def test_thresholds_table_monotonic_recall_trend_sanity() -> None:
    y_true = np.array([0, 0, 1, 1, 1], dtype=int)
    y_score = np.array([0.05, 0.10, 0.20, 0.80, 0.90], dtype=float)

    table = thresholds_table(y_true=y_true, y_score=y_score, thresholds=[0.1, 0.5, 0.9])
    assert len(table) == 3

    recalls = [row["recall"] for row in table]
    assert recalls[0] >= recalls[1] >= recalls[2]
