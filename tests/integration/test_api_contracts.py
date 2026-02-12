from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.api.app import create_app


def test_api_rejects_negative_amount(tmp_path: Path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    X = pd.DataFrame({"amount": [1, 2, 3, 10, 20, 30], "velocity_24h": [0, 1, 2, 3, 5, 8]})
    y = pd.Series([0, 0, 0, 1, 1, 1])

    model: Pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X, y)
    dump(model, artifacts / "model.joblib")

    (artifacts / "model_selection.json").write_text(
        """
{
  "objective": "expected_cost",
  "candidates": ["logreg"],
  "selected": {"kind": "logreg", "model_path": "artifacts/model.joblib"},
  "per_model": {
    "logreg": {
      "cost_report": {"best": {"threshold": 0.25, "expected_cost": 0.0, "tn": 1, "fp": 0, "fn": 0, "tp": 1}},
      "summary": {"roc_auc": 1.0, "pr_auc": 1.0}
    }
  }
}
        """.strip(),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "configs" / "dev.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        f"""
app:
  name: "fintech-antifraud-ml"
  environment: "test"

paths:
  raw_csv: "data/raw/transactions.csv"
  eval_csv: "data/processed/eval.csv"
  artifacts_dir: "{artifacts.as_posix()}"

data:
  label_column: "label"
  test_size: 0.2
  random_seed: 42
  min_rows: 10
  split_strategy: "random"
  time_column: null
  group_column: null

model:
  candidates:
    - "logreg"
  calibration:
    method: "none"
    cv: 3
  selection:
    objective: "expected_cost"

training:
  max_iter: 1000
  n_jobs: 1

evaluation:
  thresholds: [0.5]

drift:
  psi:
    warn: 0.10
    alert: 0.25
    bins: 10
    min_non_null: 10
    max_cardinality: 50
  ks:
    pvalue_warn: 0.05
    pvalue_alert: 0.01
    min_non_null: 10

cost:
  fp: 1.0
  fn: 10.0
  grid_size: 101

explainability:
  scoring: "average_precision"
  n_repeats: 5
  top_k: 10
  max_rows: 1000
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("ANTIFRAUD_CONFIG", str(cfg_path))

    app = create_app()
    client = TestClient(app)

    payload = {"records": [{"amount": -1, "velocity_24h": 1, "country": "US"}]}
    r = client.post("/score", json=payload)
    assert r.status_code == 422


def test_api_rejects_invalid_country_code_length(tmp_path: Path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    X = pd.DataFrame({"amount": [1, 2, 3, 10, 20, 30], "velocity_24h": [0, 1, 2, 3, 5, 8]})
    y = pd.Series([0, 0, 0, 1, 1, 1])

    model: Pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X, y)
    dump(model, artifacts / "model.joblib")

    (artifacts / "model_selection.json").write_text(
        """
{
  "objective": "expected_cost",
  "candidates": ["logreg"],
  "selected": {"kind": "logreg", "model_path": "artifacts/model.joblib"},
  "per_model": {
    "logreg": {
      "cost_report": {"best": {"threshold": 0.25, "expected_cost": 0.0, "tn": 1, "fp": 0, "fn": 0, "tp": 1}},
      "summary": {"roc_auc": 1.0, "pr_auc": 1.0}
    }
  }
}
        """.strip(),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "configs" / "dev.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        f"""
app:
  name: "fintech-antifraud-ml"
  environment: "test"

paths:
  raw_csv: "data/raw/transactions.csv"
  eval_csv: "data/processed/eval.csv"
  artifacts_dir: "{artifacts.as_posix()}"

data:
  label_column: "label"
  test_size: 0.2
  random_seed: 42
  min_rows: 10
  split_strategy: "random"
  time_column: null
  group_column: null

model:
  candidates:
    - "logreg"
  calibration:
    method: "none"
    cv: 3
  selection:
    objective: "expected_cost"

training:
  max_iter: 1000
  n_jobs: 1

evaluation:
  thresholds: [0.5]

drift:
  psi:
    warn: 0.10
    alert: 0.25
    bins: 10
    min_non_null: 10
    max_cardinality: 50
  ks:
    pvalue_warn: 0.05
    pvalue_alert: 0.01
    min_non_null: 10

cost:
  fp: 1.0
  fn: 10.0
  grid_size: 101

explainability:
  scoring: "average_precision"
  n_repeats: 5
  top_k: 10
  max_rows: 1000
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("ANTIFRAUD_CONFIG", str(cfg_path))

    app = create_app()
    client = TestClient(app)

    payload = {"records": [{"amount": 1, "velocity_24h": 1, "country": "USA"}]}
    r = client.post("/score", json=payload)
    assert r.status_code == 422


def test_api_accepts_payload_with_normalization(tmp_path: Path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    X = pd.DataFrame({"amount": [1, 2, 3, 10, 20, 30], "velocity_24h": [0, 1, 2, 3, 5, 8]})
    y = pd.Series([0, 0, 0, 1, 1, 1])

    model: Pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X, y)
    dump(model, artifacts / "model.joblib")

    (artifacts / "model_selection.json").write_text(
        """
{
  "objective": "expected_cost",
  "candidates": ["logreg"],
  "selected": {"kind": "logreg", "model_path": "artifacts/model.joblib"},
  "per_model": {
    "logreg": {
      "cost_report": {"best": {"threshold": 0.25, "expected_cost": 0.0, "tn": 1, "fp": 0, "fn": 0, "tp": 1}},
      "summary": {"roc_auc": 1.0, "pr_auc": 1.0}
    }
  }
}
        """.strip(),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "configs" / "dev.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        f"""
app:
  name: "fintech-antifraud-ml"
  environment: "test"

paths:
  raw_csv: "data/raw/transactions.csv"
  eval_csv: "data/processed/eval.csv"
  artifacts_dir: "{artifacts.as_posix()}"

data:
  label_column: "label"
  test_size: 0.2
  random_seed: 42
  min_rows: 10
  split_strategy: "random"
  time_column: null
  group_column: null

model:
  candidates:
    - "logreg"
  calibration:
    method: "none"
    cv: 3
  selection:
    objective: "expected_cost"

training:
  max_iter: 1000
  n_jobs: 1

evaluation:
  thresholds: [0.5]

drift:
  psi:
    warn: 0.10
    alert: 0.25
    bins: 10
    min_non_null: 10
    max_cardinality: 50
  ks:
    pvalue_warn: 0.05
    pvalue_alert: 0.01
    min_non_null: 10

cost:
  fp: 1.0
  fn: 10.0
  grid_size: 101

explainability:
  scoring: "average_precision"
  n_repeats: 5
  top_k: 10
  max_rows: 1000
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("ANTIFRAUD_CONFIG", str(cfg_path))

    app = create_app()
    client = TestClient(app)

    payload = {"records": [{"amount": 5, "velocity_24h": 1, "country": " us ", "mcc": " 54-11 ", "channel": " ECOMM "}]}
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert len(body["items"]) == 1
