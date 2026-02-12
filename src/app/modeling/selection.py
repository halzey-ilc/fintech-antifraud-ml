from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV

from app.core.config import AppConfig
from app.evaluation.cost import CostSpec, optimize_threshold_by_cost
from app.evaluation.metrics import evaluate_binary_classifier, thresholds_table
from app.modeling.train import build_model


SelectionObjective = Literal["expected_cost", "pr_auc", "roc_auc"]


@dataclass(frozen=True)
class SelectionResult:
    selected_kind: str
    selected_model_path: str
    report: dict[str, Any]


def _objective_value(objective: SelectionObjective, *, metrics: dict[str, Any]) -> float:
    if objective == "expected_cost":
        return float(metrics["cost_report"]["best"]["expected_cost"])
    if objective == "pr_auc":
        return -float(metrics["summary"]["pr_auc"])
    if objective == "roc_auc":
        return -float(metrics["summary"]["roc_auc"])
    raise ValueError(f"Unknown objective: {objective}")


def _maybe_calibrate(
    *,
    cfg: AppConfig,
    model: Any,
) -> Any:
    method = cfg.model.calibration.method
    if method == "none":
        return model
    return CalibratedClassifierCV(estimator=model, method=method, cv=cfg.model.calibration.cv)


def train_select_and_persist(
    *,
    cfg: AppConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    model_path: str,
) -> SelectionResult:
    """
    Train candidate models, evaluate on holdout, select by objective, persist the best model.

    Objective:
    - expected_cost: minimize FP/FN expected cost
    - pr_auc: maximize PR-AUC
    - roc_auc: maximize ROC-AUC
    """
    objective: SelectionObjective = cfg.model.selection.objective

    candidates = cfg.model.candidates
    per_model: dict[str, Any] = {}
    best_kind: str | None = None
    best_value: float | None = None
    best_model: Any = None

    for kind in candidates:
        base = build_model(
            kind=kind,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            max_iter=cfg.training.max_iter,
            n_jobs=cfg.training.n_jobs,
            random_seed=cfg.data.random_seed,
        )
        model = _maybe_calibrate(cfg=cfg, model=base)

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        summary = evaluate_binary_classifier(y_true=y_test, y_score=y_proba)
        thresh_table = thresholds_table(
            y_true=y_test, y_score=y_proba, thresholds=cfg.evaluation.thresholds
        )

        cost_spec = CostSpec(cost_fp=cfg.cost.fp, cost_fn=cfg.cost.fn, grid_size=cfg.cost.grid_size)
        cost_report = optimize_threshold_by_cost(y_true=y_test, y_score=y_proba, spec=cost_spec)

        metrics = {
            "kind": kind,
            "summary": summary,
            "thresholds": thresh_table,
            "cost_report": cost_report,
            "calibration": {
                "method": cfg.model.calibration.method,
                "cv": cfg.model.calibration.cv,
            },
        }
        per_model[kind] = metrics

        value = _objective_value(objective, metrics=metrics)
        if best_value is None or value < best_value:
            best_value = value
            best_kind = kind
            best_model = model

    if best_kind is None or best_model is None:
        raise RuntimeError("No model selected")

    dump(best_model, model_path)

    report = {
        "objective": objective,
        "candidates": list(candidates),
        "selected": {
            "kind": best_kind,
            "model_path": model_path,
        },
        "per_model": per_model,
    }
    return SelectionResult(selected_kind=best_kind, selected_model_path=model_path, report=report)
