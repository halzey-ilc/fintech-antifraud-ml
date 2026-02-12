from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from app.evaluation.explainability import PermutationImportanceSpec, permutation_importance_report


def test_permutation_importance_report_structure() -> None:
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [10, 11, 9, 12, 8, 13]})
    y = pd.Series([0, 0, 0, 1, 1, 1])

    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X, y)

    spec = PermutationImportanceSpec(
        scoring="average_precision",
        n_repeats=5,
        top_k=2,
        random_seed=42,
        max_rows=1000,
    )
    report = permutation_importance_report(model=model, X=X, y=y, spec=spec)

    assert report["method"] == "permutation_importance"
    assert report["top_k"] == 2
    assert "top_features" in report
    assert len(report["top_features"]) <= 2
