from __future__ import annotations

import pandas as pd

from app.observability.drift import DriftSpec, build_baseline_profile, detect_drift


def test_detect_drift_flags_alert_on_large_numeric_shift() -> None:
    spec = DriftSpec(
        bins=10,
        psi_warn=0.10,
        psi_alert=0.25,
        psi_min_non_null=10,
        psi_max_cardinality=50,
        ks_pvalue_warn=0.05,
        ks_pvalue_alert=0.01,
        ks_min_non_null=10,
    )

    X_train = pd.DataFrame(
        {
            "amount": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
            "mcc": ["a", "b"] * 50,
        }
    )
    baseline = build_baseline_profile(
        X_train, numeric_features=["amount"], categorical_features=["mcc"], spec=spec
    )

    X_eval = pd.DataFrame(
        {
            "amount": [100, 120, 130, 140, 150, 160, 170, 180, 190, 200] * 10,
            "mcc": ["a", "b"] * 50,
        }
    )

    drift = detect_drift(
        baseline,
        X_eval,
        numeric_features=["amount"],
        categorical_features=["mcc"],
        spec=spec,
    )

    assert drift["summary"]["flags"]["warn"] is True
    assert drift["summary"]["flags"]["alert"] is True
    assert "amount" in drift["summary"]["psi"]["alert_features"] or "amount" in drift["summary"]["ks"]["alert_features"]


def test_detect_drift_no_alert_on_similar_distributions() -> None:
    spec = DriftSpec(
        bins=10,
        psi_warn=0.10,
        psi_alert=0.25,
        psi_min_non_null=10,
        psi_max_cardinality=50,
        ks_pvalue_warn=0.05,
        ks_pvalue_alert=0.01,
        ks_min_non_null=10,
    )

    X_train = pd.DataFrame({"amount": [1, 2, 3, 4, 5] * 50, "mcc": ["a", "b", "c", "a", "b"] * 50})
    baseline = build_baseline_profile(
        X_train, numeric_features=["amount"], categorical_features=["mcc"], spec=spec
    )

    X_eval = pd.DataFrame({"amount": [1, 2, 3, 4, 5] * 50, "mcc": ["a", "b", "c", "a", "b"] * 50})
    drift = detect_drift(
        baseline,
        X_eval,
        numeric_features=["amount"],
        categorical_features=["mcc"],
        spec=spec,
    )

    assert drift["summary"]["flags"]["alert"] is False
