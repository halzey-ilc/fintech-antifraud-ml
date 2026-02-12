from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostSpec:
    cost_fp: float
    cost_fn: float
    grid_size: int = 201

    def __post_init__(self) -> None:
        if self.cost_fp <= 0.0:
            raise ValueError("cost_fp must be > 0")
        if self.cost_fn <= 0.0:
            raise ValueError("cost_fn must be > 0")
        if self.grid_size < 21 or self.grid_size > 5001:
            raise ValueError("grid_size must be in [21, 5001]")
        if self.grid_size % 2 == 0:
            raise ValueError("grid_size must be odd")


def _validate_inputs(y_true: pd.Series, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=int)
    if yt.ndim != 1:
        raise ValueError("y_true must be 1D")
    if not np.isin(yt, [0, 1]).all():
        raise ValueError("y_true must be binary 0/1")

    ys = np.asarray(y_score, dtype=float)
    if ys.ndim != 1:
        raise ValueError("y_score must be 1D")
    if len(ys) != len(yt):
        raise ValueError("y_true and y_score must have the same length")
    if np.isnan(ys).any():
        raise ValueError("y_score contains NaN")
    return yt, ys


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    # TN, FP, FN, TP
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def optimize_threshold_by_cost(
    *,
    y_true: pd.Series,
    y_score: np.ndarray,
    spec: CostSpec,
) -> dict[str, Any]:
    """
    Find threshold that minimizes expected cost:
      cost = FP * cost_fp + FN * cost_fn

    Returns:
      {
        "spec": {...},
        "best": {...},
        "table": [ ... per-threshold rows ... ]
      }
    """
    yt, ys = _validate_inputs(y_true, y_score)

    thresholds = np.linspace(0.0, 1.0, num=spec.grid_size, dtype=float)

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    for th in thresholds:
        y_pred = (ys >= th).astype(int)
        tn, fp, fn, tp = _confusion_counts(yt, y_pred)

        expected_cost = float(fp) * float(spec.cost_fp) + float(fn) * float(spec.cost_fn)

        row = {
            "threshold": float(th),
            "expected_cost": float(expected_cost),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        rows.append(row)

        if best_row is None or row["expected_cost"] < best_row["expected_cost"]:
            best_row = row

    if best_row is None:
        raise RuntimeError("Failed to compute best threshold")

    return {
        "spec": {
            "cost_fp": float(spec.cost_fp),
            "cost_fn": float(spec.cost_fn),
            "grid_size": int(spec.grid_size),
        },
        "best": best_row,
        "table": rows,
    }
