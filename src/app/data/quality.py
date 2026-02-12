from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import DataValidationError


@dataclass(frozen=True)
class QualityReportSpec:
    label_column: str
    max_columns_profile: int = 200
    max_unique_preview: int = 20


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        val = float(x)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except Exception:  # noqa: BLE001
        return None


def _is_bool_dtype(s: pd.Series) -> bool:
    # pandas may represent bool as bool or BooleanDtype
    return pd.api.types.is_bool_dtype(s.dtype)


def build_quality_report(df: pd.DataFrame, spec: QualityReportSpec) -> dict[str, Any]:
    if spec.label_column not in df.columns:
        raise DataValidationError(f"Label column not found: {spec.label_column}")

    rows, cols = df.shape
    if rows <= 0:
        raise DataValidationError("Dataset is empty")

    profile_cols = list(df.columns)[: spec.max_columns_profile]

    dup_rows = int(df.duplicated().sum())

    missing_rate: dict[str, float] = {}
    for col in profile_cols:
        missing_rate[col] = float(df[col].isna().mean())

    y = df[spec.label_column]
    label_counts = y.value_counts(dropna=False).to_dict()
    label_counts_str = {str(k): int(v) for k, v in label_counts.items()}
    positive_rate = float((y == 1).mean()) if pd.api.types.is_numeric_dtype(y.dtype) else None

    # Split columns by dtype:
    # - bool goes to categorical
    # - numeric excludes bool to avoid numpy percentile issues
    numeric_cols = [
        c
        for c in profile_cols
        if pd.api.types.is_numeric_dtype(df[c].dtype) and not _is_bool_dtype(df[c])
    ]
    categorical_cols = [
        c for c in profile_cols if c not in numeric_cols and c != spec.label_column
    ]

    out: dict[str, Any] = {
        "shape": {"rows": int(rows), "cols": int(cols)},
        "duplicates": {"rows": int(dup_rows)},
        "missing_rate": missing_rate,
        "label": {
            "column": spec.label_column,
            "counts": label_counts_str,
            "positive_rate": _safe_float(positive_rate),
        },
        "numeric": {},
        "categorical": {},
    }

    # Numeric profile
    for col in numeric_cols:
        s = df[col]
        non_null = s.dropna()
        if non_null.empty:
            out["numeric"][col] = {
                "non_null": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "p01": None,
                "p50": None,
                "p99": None,
            }
            continue

        # Force numeric in case of mixed dtypes
        coerced = pd.to_numeric(non_null, errors="coerce").dropna()
        if coerced.empty:
            out["numeric"][col] = {
                "non_null": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "p01": None,
                "p50": None,
                "p99": None,
            }
            continue

        out["numeric"][col] = {
            "non_null": int(coerced.shape[0]),
            "mean": _safe_float(coerced.mean()),
            "std": _safe_float(coerced.std(ddof=1)),
            "min": _safe_float(coerced.min()),
            "max": _safe_float(coerced.max()),
            "p01": _safe_float(coerced.quantile(0.01)),
            "p50": _safe_float(coerced.quantile(0.50)),
            "p99": _safe_float(coerced.quantile(0.99)),
        }

    # Categorical profile (includes bool)
    for col in categorical_cols:
        s = df[col]
        non_null = s.dropna()
        card = int(non_null.nunique(dropna=True))

        top: dict[str, int] = {}
        if not non_null.empty:
            vc = non_null.astype(str).value_counts().head(spec.max_unique_preview)
            top = {str(k): int(v) for k, v in vc.to_dict().items()}

        out["categorical"][col] = {
            "non_null": int(non_null.shape[0]),
            "cardinality": int(card),
            "top_values": top,
        }

    return out
