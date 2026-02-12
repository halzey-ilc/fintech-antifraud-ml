from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


@dataclass(frozen=True)
class PermutationImportanceSpec:
    scoring: str
    n_repeats: int
    top_k: int
    random_seed: int
    max_rows: int


def _downsample(X: pd.DataFrame, y: pd.Series, max_rows: int, random_seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    idx_sorted = np.sort(idx)
    return X.iloc[idx_sorted].reset_index(drop=True), y.iloc[idx_sorted].reset_index(drop=True)


def permutation_importance_report(
    *,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    spec: PermutationImportanceSpec,
) -> dict[str, Any]:
    """
    Permutation importance on raw input columns.
    Works with sklearn Pipelines: preprocessing is inside the model.
    """
    Xs, ys = _downsample(X, y, spec.max_rows, spec.random_seed)

    r = permutation_importance(
        model,
        Xs,
        ys,
        scoring=spec.scoring,
        n_repeats=spec.n_repeats,
        random_state=spec.random_seed,
        n_jobs=1,
    )

    cols = list(Xs.columns)
    mean = r.importances_mean.astype(float)
    std = r.importances_std.astype(float)

    order = np.argsort(-mean)
    top = order[: min(spec.top_k, len(order))]

    top_features: list[dict[str, float | str]] = []
    for i in top:
        top_features.append(
            {
                "feature": str(cols[int(i)]),
                "importance_mean": float(mean[int(i)]),
                "importance_std": float(std[int(i)]),
            }
        )

    return {
        "method": "permutation_importance",
        "scoring": spec.scoring,
        "n_repeats": int(spec.n_repeats),
        "max_rows": int(spec.max_rows),
        "top_k": int(spec.top_k),
        "rows_used": int(len(Xs)),
        "features_total": int(len(cols)),
        "top_features": top_features,
    }
