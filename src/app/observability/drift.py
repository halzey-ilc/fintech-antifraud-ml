from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


EPS = 1e-12


@dataclass(frozen=True)
class DriftSpec:
    bins: int
    psi_warn: float
    psi_alert: float
    psi_min_non_null: int
    psi_max_cardinality: int
    ks_pvalue_warn: float
    ks_pvalue_alert: float
    ks_min_non_null: int


def _psi_from_distributions(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected.astype(float), EPS, 1.0)
    a = np.clip(actual.astype(float), EPS, 1.0)
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_bins_from_reference(x_ref: pd.Series, bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(x_ref.to_numpy(), q))
    if edges.size < 3:
        edges = np.array([float(x_ref.min()), float(x_ref.max())], dtype=float)
    return edges


def _hist_share(x: pd.Series, edges: np.ndarray) -> np.ndarray:
    x_np = x.to_numpy(dtype=float)
    if edges.size >= 3:
        hist, _ = np.histogram(x_np, bins=edges)
    else:
        hist = np.array([x_np.size], dtype=int)
    share = hist / max(int(np.sum(hist)), 1)
    return share.astype(float)


def _categorical_share(s: pd.Series) -> dict[str, float]:
    v = s.dropna().astype(str)
    if v.empty:
        return {}
    counts = v.value_counts(normalize=True)
    return {str(k): float(vv) for k, vv in counts.to_dict().items()}


def build_baseline_profile(
    X_train: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    spec: DriftSpec,
) -> dict[str, Any]:
    """
    Baseline profile computed on training features (raw, before preprocessing).
    Stores:
    - numeric: bin edges + reference distribution + reference sample size
    - categorical: normalized value frequencies (possibly truncated by cardinality)
    """
    profile: dict[str, Any] = {
        "numeric": {},
        "categorical": {},
        "meta": {
            "bins": spec.bins,
        },
    }

    for col in numeric_features:
        if col not in X_train.columns:
            continue
        x_ref = pd.to_numeric(X_train[col], errors="coerce").dropna()
        if int(x_ref.shape[0]) < spec.psi_min_non_null:
            continue
        edges = _numeric_bins_from_reference(x_ref, spec.bins)
        share = _hist_share(x_ref, edges)
        profile["numeric"][col] = {
            "edges": [float(e) for e in edges.tolist()],
            "ref_share": [float(v) for v in share.tolist()],
            "ref_non_null": int(x_ref.shape[0]),
        }

    for col in categorical_features:
        if col not in X_train.columns:
            continue
        v = X_train[col].dropna().astype(str)
        if int(v.shape[0]) < spec.psi_min_non_null:
            continue
        card = int(v.nunique(dropna=True))
        if card > spec.psi_max_cardinality:
            top = v.value_counts().head(spec.psi_max_cardinality).index
            v = v[v.isin(top)]
        freq = _categorical_share(v)
        if not freq:
            continue
        profile["categorical"][col] = {
            "ref_freq": freq,
            "ref_non_null": int(v.shape[0]),
            "ref_cardinality": int(len(freq)),
        }

    return profile


def _psi_numeric(col_profile: dict[str, Any], x_actual: pd.Series) -> float | None:
    edges = np.asarray(col_profile["edges"], dtype=float)
    ref_share = np.asarray(col_profile["ref_share"], dtype=float)

    xa = pd.to_numeric(x_actual, errors="coerce").dropna()
    if xa.empty:
        return None
    act_share = _hist_share(xa, edges)
    if act_share.size != ref_share.size:
        return None
    return _psi_from_distributions(ref_share, act_share)


def _psi_categorical(col_profile: dict[str, Any], s_actual: pd.Series) -> float | None:
    ref_freq: dict[str, float] = col_profile["ref_freq"]
    act_freq = _categorical_share(s_actual)

    keys = set(ref_freq.keys()) | set(act_freq.keys())
    if not keys:
        return None

    expected = np.array([ref_freq.get(k, 0.0) for k in keys], dtype=float)
    actual = np.array([act_freq.get(k, 0.0) for k in keys], dtype=float)

    expected = expected / max(float(np.sum(expected)), EPS)
    actual = actual / max(float(np.sum(actual)), EPS)
    return _psi_from_distributions(expected, actual)


def detect_drift(
    baseline_profile: dict[str, Any],
    X_eval: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    spec: DriftSpec,
) -> dict[str, Any]:
    """
    Compute drift metrics between baseline and evaluation data.

    Outputs:
    - per-feature metrics
    - global flags: warn/alert based on thresholds
    """
    numeric_out: dict[str, Any] = {}
    categorical_out: dict[str, Any] = {}

    psi_alert_features: list[str] = []
    psi_warn_features: list[str] = []
    ks_alert_features: list[str] = []
    ks_warn_features: list[str] = []

    baseline_num = baseline_profile.get("numeric", {})
    for col in numeric_features:
        if col not in X_eval.columns:
            continue
        if col not in baseline_num:
            continue

        psi = _psi_numeric(baseline_num[col], X_eval[col])
        x_ref_n = int(baseline_num[col].get("ref_non_null", 0))
        x_eval_n = int(pd.to_numeric(X_eval[col], errors="coerce").dropna().shape[0])

        ks_p = None
        if x_ref_n >= spec.ks_min_non_null and x_eval_n >= spec.ks_min_non_null:
            x_eval = pd.to_numeric(X_eval[col], errors="coerce").dropna().to_numpy(dtype=float)
            edges = np.asarray(baseline_num[col]["edges"], dtype=float)
            ref_share = np.asarray(baseline_num[col]["ref_share"], dtype=float)
            ref_centers = (edges[:-1] + edges[1:]) / 2.0 if edges.size >= 3 else np.array([0.0])
            ref_counts = np.maximum((ref_share * x_ref_n).astype(int), 1)
            x_ref_approx = np.repeat(ref_centers, ref_counts)
            ks_p = float(ks_2samp(x_ref_approx, x_eval, alternative="two-sided", mode="auto").pvalue)

        numeric_out[col] = {
            "psi": None if psi is None else float(psi),
            "psi_ref_non_null": x_ref_n,
            "psi_eval_non_null": x_eval_n,
            "ks_pvalue": ks_p,
        }

        if psi is not None:
            if psi >= spec.psi_alert:
                psi_alert_features.append(col)
            elif psi >= spec.psi_warn:
                psi_warn_features.append(col)

        if ks_p is not None:
            if ks_p <= spec.ks_pvalue_alert:
                ks_alert_features.append(col)
            elif ks_p <= spec.ks_pvalue_warn:
                ks_warn_features.append(col)

    baseline_cat = baseline_profile.get("categorical", {})
    for col in categorical_features:
        if col not in X_eval.columns:
            continue
        if col not in baseline_cat:
            continue

        psi = _psi_categorical(baseline_cat[col], X_eval[col])
        x_ref_n = int(baseline_cat[col].get("ref_non_null", 0))
        x_eval_n = int(X_eval[col].dropna().shape[0])

        categorical_out[col] = {
            "psi": None if psi is None else float(psi),
            "psi_ref_non_null": x_ref_n,
            "psi_eval_non_null": x_eval_n,
            "ref_cardinality": int(baseline_cat[col].get("ref_cardinality", 0)),
        }

        if psi is not None:
            if psi >= spec.psi_alert:
                psi_alert_features.append(col)
            elif psi >= spec.psi_warn:
                psi_warn_features.append(col)

    summary = {
        "psi": {
            "warn_threshold": spec.psi_warn,
            "alert_threshold": spec.psi_alert,
            "warn_features": sorted(set(psi_warn_features)),
            "alert_features": sorted(set(psi_alert_features)),
            "warn_count": int(len(set(psi_warn_features))),
            "alert_count": int(len(set(psi_alert_features))),
        },
        "ks": {
            "pvalue_warn": spec.ks_pvalue_warn,
            "pvalue_alert": spec.ks_pvalue_alert,
            "warn_features": sorted(set(ks_warn_features)),
            "alert_features": sorted(set(ks_alert_features)),
            "warn_count": int(len(set(ks_warn_features))),
            "alert_count": int(len(set(ks_alert_features))),
        },
        "flags": {
            "warn": bool(set(psi_warn_features) or set(ks_warn_features)),
            "alert": bool(set(psi_alert_features) or set(ks_alert_features)),
        },
    }

    return {
        "summary": summary,
        "numeric": numeric_out,
        "categorical": categorical_out,
    }
