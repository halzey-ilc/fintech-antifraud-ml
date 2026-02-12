from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd


@dataclass(frozen=True)
class FeatureInfo:
    numeric_features: list[str]
    categorical_features: list[str]


LABEL_DEFAULT: Final[str] = "label"


def build_feature_matrix(df: pd.DataFrame, label_col: str = LABEL_DEFAULT) -> tuple[pd.DataFrame, pd.Series, FeatureInfo]:
    """
    Build feature matrix X and label vector y.

    Convention:
    - label is binary 0/1 in column `label_col`
    - numeric features are int/float columns excluding label
    - categorical features are object/string columns excluding label

    For real fintech data, this module will evolve into:
    - time-window aggregations
    - entity-based statistics
    - leakage guards (no future information)
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    feature_info = FeatureInfo(numeric_features=sorted(numeric_cols), categorical_features=sorted(categorical_cols))
    return X, y, feature_info
