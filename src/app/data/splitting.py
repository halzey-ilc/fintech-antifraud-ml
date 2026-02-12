from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from app.core.exceptions import DataValidationError


@dataclass(frozen=True)
class SplitSpec:
    test_size: float
    random_seed: int
    strategy: str  # "random" | "time"
    label_column: str
    time_column: str | None = None
    group_column: str | None = None


@dataclass(frozen=True)
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metadata: dict[str, Any]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def _parse_time_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_datetime(df[col], errors="coerce", utc=True)
    if s.isna().any():
        bad = int(s.isna().sum())
        raise DataValidationError(f"Time column '{col}' contains {bad} unparsable values")
    return s


def split_dataset(df: pd.DataFrame, spec: SplitSpec) -> SplitResult:
    """
    Split a labeled dataset into train/test with leakage guards.

    Strategies:
    - random:
        - stratified by label
        - if group_column provided: GroupShuffleSplit (no group overlap)
    - time:
        - chronological split by time_column
        - if group_column provided: split by group max time (no group overlap, time-respecting)
    """
    if df.empty:
        raise DataValidationError("Dataset is empty")

    _require_columns(df, [spec.label_column])

    y = df[spec.label_column].astype(int)
    X = df.drop(columns=[spec.label_column])

    unique = set(y.unique().tolist())
    if not unique.issubset({0, 1}):
        raise DataValidationError(f"Label column must be binary 0/1. Got: {sorted(unique)}")

    if spec.strategy == "random":
        if spec.group_column:
            _require_columns(df, [spec.group_column])
            groups = df[spec.group_column]
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=spec.test_size,
                random_state=spec.random_seed,
            )
            idx = np.arange(len(df))
            train_idx, test_idx = next(gss.split(idx, y, groups=groups))
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_test = y.iloc[test_idx].reset_index(drop=True)

            meta = {
                "strategy": "random_group",
                "group_column": spec.group_column,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "train_pos_rate": float(y_train.mean()) if len(y_train) else 0.0,
                "test_pos_rate": float(y_test.mean()) if len(y_test) else 0.0,
            }
            return SplitResult(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, metadata=meta)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=spec.test_size,
            random_state=spec.random_seed,
            stratify=y,
        )
        meta = {
            "strategy": "random_stratified",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_pos_rate": float(y_train.mean()) if len(y_train) else 0.0,
            "test_pos_rate": float(y_test.mean()) if len(y_test) else 0.0,
        }
        return SplitResult(
            X_train=X_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            metadata=meta,
        )

    if spec.strategy == "time":
        if not spec.time_column:
            raise DataValidationError("time_column must be set for time strategy")

        _require_columns(df, [spec.time_column])
        t = _parse_time_series(df, spec.time_column)

        if spec.group_column:
            _require_columns(df, [spec.group_column])

            tmp = pd.DataFrame(
                {
                    "group": df[spec.group_column],
                    "time": t,
                }
            )
            group_max = tmp.groupby("group", sort=False)["time"].max().sort_values()
            groups_sorted = group_max.index.to_list()

            target_test_rows = max(1, int(round(len(df) * spec.test_size)))
            test_groups: list[Any] = []
            test_rows = 0

            group_to_count = df[spec.group_column].value_counts().to_dict()
            for g in reversed(groups_sorted):
                test_groups.append(g)
                test_rows += int(group_to_count.get(g, 0))
                if test_rows >= target_test_rows:
                    break

            test_group_set = set(test_groups)
            is_test = df[spec.group_column].isin(test_group_set)

            df_train = df.loc[~is_test].copy()
            df_test = df.loc[is_test].copy()

            if df_train.empty or df_test.empty:
                raise DataValidationError("Time+group split produced empty train or test set")

            X_train = df_train.drop(columns=[spec.label_column]).reset_index(drop=True)
            y_train = df_train[spec.label_column].astype(int).reset_index(drop=True)
            X_test = df_test.drop(columns=[spec.label_column]).reset_index(drop=True)
            y_test = df_test[spec.label_column].astype(int).reset_index(drop=True)

            meta = {
                "strategy": "time_group",
                "time_column": spec.time_column,
                "group_column": spec.group_column,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "train_pos_rate": float(y_train.mean()) if len(y_train) else 0.0,
                "test_pos_rate": float(y_test.mean()) if len(y_test) else 0.0,
                "test_groups": int(len(test_group_set)),
            }
            return SplitResult(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, metadata=meta)

        order = t.sort_values().index
        df_sorted = df.loc[order].reset_index(drop=True)

        n = len(df_sorted)
        cutoff = int(np.floor(n * (1.0 - spec.test_size)))
        cutoff = min(max(cutoff, 1), n - 1)

        df_train = df_sorted.iloc[:cutoff].copy()
        df_test = df_sorted.iloc[cutoff:].copy()

        X_train = df_train.drop(columns=[spec.label_column]).reset_index(drop=True)
        y_train = df_train[spec.label_column].astype(int).reset_index(drop=True)
        X_test = df_test.drop(columns=[spec.label_column]).reset_index(drop=True)
        y_test = df_test[spec.label_column].astype(int).reset_index(drop=True)

        meta = {
            "strategy": "time",
            "time_column": spec.time_column,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_pos_rate": float(y_train.mean()) if len(y_train) else 0.0,
            "test_pos_rate": float(y_test.mean()) if len(y_test) else 0.0,
            "cutoff_index": int(cutoff),
        }
        return SplitResult(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, metadata=meta)

    raise DataValidationError(f"Unknown split strategy: {spec.strategy}")
