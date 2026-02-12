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
from app.data.splitting import SplitSpec, split_dataset
from app.evaluation.explainability import PermutationImportanceSpec, permutation_importance_report
from app.features.transactions import build_feature_matrix
from app.modeling.selection import train_select_and_persist
from app.observability.drift import DriftSpec, build_baseline_profile
from app.pipeline.steps import PipelineResult
from app.storage.local import LocalArtifactStore
from app.utils.hash import sha256_file
from app.utils.run_id import RunId


log = get_logger("pipeline.train")


@dataclass(frozen=True)
class TrainArtifacts:
    model_rel: str = "model.joblib"
    metrics_rel: str = "metrics.json"
    dataset_fingerprint_rel: str = "dataset_fingerprint.json"
    run_metadata_rel: str = "run_metadata.json"
    baseline_profile_rel: str = "baseline_profile.json"
    quality_train_rel: str = "data_quality_train.json"
    model_selection_rel: str = "model_selection.json"
    explainability_train_rel: str = "explainability_train.json"


def _validate_dataframe(df: pd.DataFrame, label_col: str, min_rows: int) -> None:
    if df.empty:
        raise DataValidationError("Dataset is empty")

    if len(df) < min_rows:
        raise DataValidationError(f"Dataset has too few rows: {len(df)} < {min_rows}")

    if label_col not in df.columns:
        raise DataValidationError(f"Missing label column: {label_col}")

    if df[label_col].isna().any():
        raise DataValidationError("Label column contains missing values")

    unique = set(df[label_col].unique().tolist())
    if not unique.issubset({0, 1}):
        raise DataValidationError(f"Label column must be binary 0/1. Got: {sorted(unique)}")


def run_train(cfg: AppConfig, paths: ResolvedPaths, project_root: Path) -> PipelineResult:
    ensure_directories(paths)

    if not paths.raw_csv.exists():
        raise DataValidationError(f"Raw CSV not found: {paths.raw_csv}")

    log.info("Reading dataset: %s", paths.raw_csv)
    df = pd.read_csv(paths.raw_csv)

    _validate_dataframe(df, cfg.data.label_column, cfg.data.min_rows)

    _, _, feature_info = build_feature_matrix(df, label_col=cfg.data.label_column)

    split_spec = SplitSpec(
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
        strategy=cfg.data.split_strategy,
        label_column=cfg.data.label_column,
        time_column=cfg.data.time_column,
        group_column=cfg.data.group_column,
    )
    split_res = split_dataset(df, split_spec)
    X_train, X_test = split_res.X_train, split_res.X_test
    y_train, y_test = split_res.y_train, split_res.y_test

    store = LocalArtifactStore(paths.artifacts_dir)
    run_id = RunId.new("train").value
    layout = store.prepare_run(run_id)
    artifacts = TrainArtifacts()

    log.info("Run created: %s", run_id)

    quality = build_quality_report(df, QualityReportSpec(label_column=cfg.data.label_column))
    quality_file = store.write_json(layout.run_dir, artifacts.quality_train_rel, quality)

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
    baseline_profile = build_baseline_profile(
        X_train,
        numeric_features=feature_info.numeric_features,
        categorical_features=feature_info.categorical_features,
        spec=drift_spec,
    )
    baseline_file = store.write_json(layout.run_dir, artifacts.baseline_profile_rel, baseline_profile)

    model_path = (layout.run_dir / artifacts.model_rel).resolve()

    log.info("Training candidates=%s objective=%s", cfg.model.candidates, cfg.model.selection.objective)
    sel = train_select_and_persist(
        cfg=cfg,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        numeric_features=feature_info.numeric_features,
        categorical_features=feature_info.categorical_features,
        model_path=str(model_path),
    )
    selection_file = store.write_json(layout.run_dir, artifacts.model_selection_rel, sel.report)

    model = load(model_path)

    explain_spec = PermutationImportanceSpec(
        scoring=cfg.explainability.scoring,
        n_repeats=cfg.explainability.n_repeats,
        top_k=cfg.explainability.top_k,
        random_seed=cfg.data.random_seed,
        max_rows=cfg.explainability.max_rows,
    )
    explain = permutation_importance_report(model=model, X=X_test, y=y_test, spec=explain_spec)
    explain_payload: dict[str, Any] = {
        "selected_model": {"kind": sel.selected_kind, "model_path": str(model_path)},
        "split": split_res.metadata,
        "report": explain,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    explain_file = store.write_json(layout.run_dir, artifacts.explainability_train_rel, explain_payload)

    metrics_payload: dict[str, Any] = {
        "run_id": run_id,
        "selected_model": {"kind": sel.selected_kind, "model_path": str(model_path)},
        "model_selection_report_path": str(selection_file),
        "explainability_report_path": str(explain_file),
        "split": split_res.metadata,
        "data": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "label_column": cfg.data.label_column,
        },
        "quality_report_path": str(quality_file),
        "baseline_profile_path": str(baseline_file),
    }
    metrics_file = store.write_json(layout.run_dir, artifacts.metrics_rel, metrics_payload)

    fingerprint_payload = {
        "raw_csv": str(paths.raw_csv),
        "raw_csv_sha256": sha256_file(paths.raw_csv),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    fingerprint_file = store.write_json(layout.run_dir, artifacts.dataset_fingerprint_rel, fingerprint_payload)

    run_metadata_payload = {
        "run_id": run_id,
        "run_type": "train",
        "app": cfg.app.model_dump(),
        "config_snapshot": cfg.model_dump(),
        "project_root": str(project_root.resolve()),
        "artifacts_dir": str(paths.artifacts_dir.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_metadata_file = store.write_json(layout.run_dir, artifacts.run_metadata_rel, run_metadata_payload)

    # Update "latest" copies for stable consumers (API, humans).
    store.copy_to_latest(
        layout,
        files=[
            model_path,
            selection_file,
            metrics_file,
            baseline_file,
            quality_file,
            explain_file,
            run_metadata_file,
        ],
    )
    store.update_latest_pointer(
        layout,
        payload={
            "run_type": "train",
            "created_utc": run_metadata_payload["created_utc"],
            "selected_model_kind": sel.selected_kind,
            "run_dir": str(layout.run_dir),
        },
    )

    log.info("Artifacts saved to run: %s", layout.run_dir)
    return PipelineResult(
        model_path=str(model_path),
        metrics_path=str(metrics_file),
        dataset_fingerprint_path=str(fingerprint_file),
        run_metadata_path=str(run_metadata_file),
    )




"""

/// This come mostly from the original codebase ///


def run_train(cfg: AppConfig, paths: ResolvedPaths, project_root: Path) -> PipelineResult:
    ensure_directories(paths)

    if not paths.raw_csv.exists():
        raise DataValidationError(f"Raw CSV not found: {paths.raw_csv}")

    log.info("Reading dataset: %s", paths.raw_csv)
    df = pd.read_csv(paths.raw_csv)

    _validate_dataframe(df, cfg.data.label_column, cfg.data.min_rows)

    _, _, feature_info = build_feature_matrix(df, label_col=cfg.data.label_column)

    split_spec = SplitSpec(
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
        strategy=cfg.data.split_strategy,
        label_column=cfg.data.label_column,
        time_column=cfg.data.time_column,
        group_column=cfg.data.group_column,
    )

    split_res = split_dataset(df, split_spec)
    X_train, X_test = split_res.X_train, split_res.X_test
    y_train, y_test = split_res.y_train, split_res.y_test

    log.info("Split strategy: %s", split_res.metadata.get("strategy"))

    store = LocalArtifactStore(paths.artifacts_dir)
    artifacts = TrainArtifacts()

    quality = build_quality_report(df, QualityReportSpec(label_column=cfg.data.label_column))
    quality_file = store.write_json(artifacts.quality_train_rel, quality)

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
    baseline_profile = build_baseline_profile(
        X_train,
        numeric_features=feature_info.numeric_features,
        categorical_features=feature_info.categorical_features,
        spec=drift_spec,
    )
    baseline_file = store.write_json(artifacts.baseline_profile_rel, baseline_profile)

    model_path = (paths.artifacts_dir / artifacts.model_rel).resolve()

    log.info("Training candidates=%s objective=%s", cfg.model.candidates, cfg.model.selection.objective)
    sel = train_select_and_persist(
        cfg=cfg,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        numeric_features=feature_info.numeric_features,
        categorical_features=feature_info.categorical_features,
        model_path=str(model_path),
    )
    selection_file = store.write_json(artifacts.model_selection_rel, sel.report)

    metrics_payload: dict[str, Any] = {
        "selected_model": {
            "kind": sel.selected_kind,
            "model_path": sel.selected_model_path,
        },
        "model_selection_report_path": str(selection_file),
        "split": split_res.metadata,
        "data": {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "label_column": cfg.data.label_column,
        },
        "quality_report_path": str(quality_file),
        "baseline_profile_path": str(baseline_file),
    }
    metrics_file = store.write_json(artifacts.metrics_rel, metrics_payload)

    fingerprint_payload = {
        "raw_csv": str(paths.raw_csv),
        "raw_csv_sha256": sha256_file(paths.raw_csv),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    fingerprint_file = store.write_json(artifacts.dataset_fingerprint_rel, fingerprint_payload)

    run_metadata_payload = {
        "app": cfg.app.model_dump(),
        "config_snapshot": cfg.model_dump(),
        "project_root": str(project_root.resolve()),
        "artifacts_dir": str(paths.artifacts_dir.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_metadata_file = store.write_json(artifacts.run_metadata_rel, run_metadata_payload)

    log.info("Artifacts saved to: %s", paths.artifacts_dir)
    return PipelineResult(
        model_path=str(model_path),
        metrics_path=str(metrics_file),
        dataset_fingerprint_path=str(fingerprint_file),
        run_metadata_path=str(run_metadata_file),
    )

"""