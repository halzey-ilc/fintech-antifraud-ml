from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import AppConfig
from app.modeling.selection import train_select_and_persist


def _cfg_single(kind: str) -> AppConfig:
    payload = {
        "app": {"name": "x", "environment": "test"},
        "paths": {
            "raw_csv": "data/raw/transactions.csv",
            "eval_csv": "data/processed/eval.csv",
            "artifacts_dir": "artifacts",
        },
        "data": {
            "label_column": "label",
            "test_size": 0.2,
            "random_seed": 42,
            "min_rows": 10,
            "split_strategy": "random",
            "time_column": None,
            "group_column": None,
        },
        "model": {
            "candidates": [kind],
            "calibration": {"method": "none", "cv": 3},
            "selection": {"objective": "expected_cost"},
        },
        "training": {"max_iter": 1000, "n_jobs": 1},
        "evaluation": {"thresholds": [0.2, 0.5, 0.8]},
        "drift": {
            "psi": {"warn": 0.1, "alert": 0.25, "bins": 10, "min_non_null": 10, "max_cardinality": 50},
            "ks": {"pvalue_warn": 0.05, "pvalue_alert": 0.01, "min_non_null": 10},
        },
        "cost": {"fp": 1.0, "fn": 10.0, "grid_size": 101},
    }
    return AppConfig.model_validate(payload)


def test_train_select_single_candidate_persists_model(tmp_path: Path) -> None:
    cfg = _cfg_single("logreg")
    X_train = pd.DataFrame({"amount": [1, 2, 3, 4, 5, 6], "mcc": ["a", "b", "a", "b", "a", "b"]})
    y_train = pd.Series([0, 0, 0, 1, 1, 1], name="label")
    X_test = pd.DataFrame({"amount": [2, 3, 7, 8], "mcc": ["a", "b", "a", "b"]})
    y_test = pd.Series([0, 0, 1, 1], name="label")

    model_path = tmp_path / "model.joblib"
    res = train_select_and_persist(
        cfg=cfg,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        numeric_features=["amount"],
        categorical_features=["mcc"],
        model_path=str(model_path),
    )

    assert res.selected_kind == "logreg"
    assert model_path.exists()
    assert "per_model" in res.report
    assert "logreg" in res.report["per_model"]
