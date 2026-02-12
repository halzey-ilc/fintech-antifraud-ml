from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load

from app.core.config import AppConfig, ResolvedPaths, ensure_directories
from app.core.exceptions import DataValidationError
from app.core.logging import get_logger
from app.data.quality import QualityReportSpec, build_quality_report
from app.evaluation.cost import CostSpec, optimize_threshold_by_cost
from app.evaluation.explainability import PermutationImportanceSpec, permutation_importance_report
from app.evaluation.metrics import evaluate_binary_classifier, thresholds_table
from app.features.transactions import build_feature_matrix
from app.observability.drift import DriftSpec, detect_drift
from app.storage.local import LocalArtifactStore
from app.utils.hash import sha256_file
from app.utils.run_id import RunId


log = get_logger("pipeline.eval")


@dataclass(frozen=True)
class EvalArtifacts:
    eval_metrics_rel: str = "eval_metrics.json"
    eval_run_metadata_rel: str = "eval_run_metadata.json"
    quality_eval_rel: str = "data_quality_eval.json"
    drift_eval_rel: str = "drift_eval.json"
    explainability_eval_rel: str = "explainability_eval.json"


def _validate_eval_dataframe(df: pd.DataFrame, label_col: str) -> None:
    if df.empty:
        raise DataValidationError("Eval dataset is empty")

    if label_col not in df.columns:
        raise DataValidationError(f"Missing label column in eval dataset: {label_col}")

    if df[label_col].isna().any():
        raise DataValidationError("Label column in eval dataset contains missing values")

    unique = set(df[label_col].unique().tolist())
    if not unique.issubset({0, 1}):
        raise DataValidationError(f"Eval label must be binary 0/1. Got: {sorted(unique)}")


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def run_eval(cfg: AppConfig, paths: ResolvedPaths, project_root: Path) -> dict[str, str]:
    ensure_directories(paths)

    # Prefer latest model artifacts; fallback is handled by model_path existence checks.
    model_path = (paths.artifacts_dir / "latest" / "model.joblib").resolve()
    if not model_path.exists():
        model_path = (paths.artifacts_dir / "model.joblib").resolve()
    if not model_path.exists():
        raise DataValidationError(f"Trained model not found: {model_path}. Run train first.")

    baseline_path = (paths.artifacts_dir / "latest" / "baseline_profile.json").resolve()
    if not baseline_path.exists():
        baseline_path = (paths.artifacts_dir / "baseline_profile.json").resolve()
    if not baseline_path.exists():
        raise DataValidationError(f"Baseline profile not found: {baseline_path}. Run train first.")

    if not paths.eval_csv.exists():
        raise DataValidationError(f"Eval CSV not found: {paths.eval_csv}")

    log.info("Loading model: %s", model_path)
    model = load(model_path)

    log.info("Reading eval dataset: %s", paths.eval_csv)
    df_eval = pd.read_csv(paths.eval_csv)

    _validate_eval_dataframe(df_eval, cfg.data.label_column)

    X_eval, y_eval, feature_info = build_feature_matrix(df_eval, label_col=cfg.data.label_column)

    store = LocalArtifactStore(paths.artifacts_dir)
    run_id = RunId.new("eval").value
    layout = store.prepare_run(run_id)
    artifacts = EvalArtifacts()

    quality = build_quality_report(df_eval, QualityReportSpec(label_column=cfg.data.label_column))
    quality_file = store.write_json(layout.run_dir, artifacts.quality_eval_rel, quality)

    baseline_profile = json_load(baseline_path)

    drift_spec = DriftSpec(
        bins=cfg.drift.psi.bins,
        psi_warn=cfg.drift.psi.warn,
        psi_alert=cfg.drift.psi.alert,
        psi_min_non_null=cfg.drift.psi.min_non_null,
        psi_max_cardinality=cfg.drift.psi.max_cardinality,
        ks_pvalue_warn=cfg.drift.ks.pvalue_warn,
        ks_pvalue_alert=cfg.drift.ks.pvalue_alert,
        ks_min_non_null=cfg.drift.ks.min_non_null,
    )
    drift = detect_drift(
        baseline_profile,
        X_eval,
        numeric_features=feature_info.numeric_features,
        categorical_features=feature_info.categorical_features,
        spec=drift_spec,
    )
    drift_payload: dict[str, Any] = {
        "summary": drift["summary"],
        "numeric": drift["numeric"],
        "categorical": drift["categorical"],
        "baseline_profile_path": str(baseline_path),
        "eval_csv": str(paths.eval_csv),
        "eval_csv_sha256": sha256_file(paths.eval_csv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    drift_file = store.write_json(layout.run_dir, artifacts.drift_eval_rel, drift_payload)

    explain_spec = PermutationImportanceSpec(
        scoring=cfg.explainability.scoring,
        n_repeats=cfg.explainability.n_repeats,
        top_k=cfg.explainability.top_k,
        random_seed=cfg.data.random_seed,
        max_rows=cfg.explainability.max_rows,
    )
    explain = permutation_importance_report(model=model, X=X_eval, y=y_eval, spec=explain_spec)
    explain_payload: dict[str, Any] = {
        "model_path": str(model_path),
        "eval_csv": str(paths.eval_csv),
        "report": explain,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    explain_file = store.write_json(layout.run_dir, artifacts.explainability_eval_rel, explain_payload)

    log.info("Scoring eval dataset")
    y_proba = model.predict_proba(X_eval)[:, 1]

    summary = evaluate_binary_classifier(y_true=y_eval, y_score=y_proba)
    table = thresholds_table(y_true=y_eval, y_score=y_proba, thresholds=cfg.evaluation.thresholds)

    cost_spec = CostSpec(cost_fp=cfg.cost.fp, cost_fn=cfg.cost.fn, grid_size=cfg.cost.grid_size)
    cost_report = optimize_threshold_by_cost(y_true=y_eval, y_score=y_proba, spec=cost_spec)

    metrics_payload: dict[str, Any] = {
        "run_id": run_id,
        "summary": summary,
        "thresholds": table,
        "cost_report": cost_report,
        "eval_data": {
            "path": str(paths.eval_csv),
            "sha256": sha256_file(paths.eval_csv),
            "rows": int(df_eval.shape[0]),
            "cols": int(df_eval.shape[1]),
            "label_column": cfg.data.label_column,
        },
        "model": {"path": str(model_path)},
        "quality_report_path": str(quality_file),
        "drift_report_path": str(drift_file),
        "explainability_report_path": str(explain_file),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metrics_file = store.write_json(layout.run_dir, artifacts.eval_metrics_rel, metrics_payload)

    run_metadata_payload = {
        "run_id": run_id,
        "run_type": "eval",
        "app": cfg.app.model_dump(),
        "config_snapshot": cfg.model_dump(),
        "project_root": str(project_root.resolve()),
        "artifacts_dir": str(paths.artifacts_dir.resolve()),
        "eval_csv": str(paths.eval_csv),
        "eval_csv_sha256": sha256_file(paths.eval_csv),
        "model_path_used": str(model_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_metadata_file = store.write_json(layout.run_dir, artifacts.eval_run_metadata_rel, run_metadata_payload)

    # Update latest pointer for convenience (optional; keeps last eval run visible too).
    store.copy_to_latest(
        layout,
        files=[
            metrics_file,
            drift_file,
            quality_file,
            explain_file,
            run_metadata_file,
        ],
    )
    store.update_latest_pointer(
        layout,
        payload={
            "run_type": "eval",
            "created_utc": run_metadata_payload["created_utc"],
            "run_dir": str(layout.run_dir),
        },
    )

    log.info("Eval artifacts saved to run: %s", layout.run_dir)
    return {
        "eval_metrics_path": str(metrics_file),
        "eval_run_metadata_path": str(run_metadata_file),
        "drift_report_path": str(drift_file),
        "quality_report_path": str(quality_file),
        "explainability_report_path": str(explain_file),
    }
#  how to set and with search for it in the config file for example data.split = "data.txt" statement suit you  