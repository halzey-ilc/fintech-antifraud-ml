from __future__ import annotations

from pathlib import Path

from app.core.config import AppConfig, resolve_paths


def test_resolve_paths_defaults_eval_csv(tmp_path: Path) -> None:
    cfg_payload = {
        "app": {"name": "x", "environment": "test"},
        "paths": {
            "raw_csv": "data/raw/transactions.csv",
            "artifacts_dir": "artifacts",
        },
        "data": {
            "label_column": "label",
            "test_size": 0.2,
            "random_seed": 42,
            "min_rows": 10,
        },
        "model": {"kind": "logreg", "threshold": 0.5},
        "training": {"max_iter": 1000, "n_jobs": 1},
        "evaluation": {"thresholds": [0.5]},
    }

    cfg = AppConfig.model_validate(cfg_payload)
    paths = resolve_paths(cfg, tmp_path)

    assert paths.raw_csv == (tmp_path / "data/raw/transactions.csv").resolve()
    assert paths.artifacts_dir == (tmp_path / "artifacts").resolve()
    assert paths.eval_csv == (tmp_path / "data/processed/eval.csv").resolve()
