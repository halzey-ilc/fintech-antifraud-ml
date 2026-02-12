from __future__ import annotations

import numpy as np

from app.evaluation.cost import CostSpec, optimize_threshold_by_cost


def test_optimize_threshold_prefers_high_recall_when_fn_cost_high() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    y_score = np.array([0.10, 0.20, 0.30, 0.40, 0.41, 0.42, 0.43, 0.44], dtype=float)

    spec = CostSpec(cost_fp=1.0, cost_fn=100.0, grid_size=101)
    report = optimize_threshold_by_cost(y_true=y_true, y_score=y_score, spec=spec)

    # With very high FN cost, threshold should be low enough to capture positives.
    assert report["best"]["threshold"] <= 0.44


def test_optimize_threshold_prefers_precision_when_fp_cost_high() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1], dtype=int)
    y_score = np.array([0.60, 0.55, 0.50, 0.45, 0.40, 0.35], dtype=float)

    spec = CostSpec(cost_fp=100.0, cost_fn=1.0, grid_size=101)
    report = optimize_threshold_by_cost(y_true=y_true, y_score=y_score, spec=spec)

    # With very high FP cost, threshold should be relatively high.
    assert report["best"]["threshold"] >= 0.45
