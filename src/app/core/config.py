from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class AppSection(BaseModel):
    name: str
    environment: str


class PathsSection(BaseModel):
    raw_csv: str
    artifacts_dir: str
    eval_csv: str = "data/processed/eval.csv"


class DataSection(BaseModel):
    label_column: str
    test_size: float
    random_seed: int
    min_rows: int

    split_strategy: str = "random"
    time_column: str | None = None
    group_column: str | None = None


class CalibrationSection(BaseModel):
    method: str = "none"
    cv: int = 3


class SelectionSection(BaseModel):
    objective: str = "expected_cost"


class ModelSection(BaseModel):
    """
    Backward/forward compatible model section.

    Supports:
      - legacy: {"kind": "...", "threshold": 0.5}
      - current: {"candidates": [...], "calibration": {...}, "selection": {...}}
    """

    model_config = ConfigDict(extra="allow")

    # Legacy
    kind: str | None = None
    threshold: float | None = None

    # Current
    candidates: list[str] = Field(default_factory=list)
    calibration: CalibrationSection = Field(default_factory=CalibrationSection)
    selection: SelectionSection = Field(default_factory=SelectionSection)


class TrainingSection(BaseModel):
    max_iter: int = 1000
    n_jobs: int | None = None


class EvaluationSection(BaseModel):
    thresholds: list[float] = Field(default_factory=lambda: [0.5])


class DriftPsiSection(BaseModel):
    warn: float = 0.10
    alert: float = 0.25
    bins: int = 10
    min_non_null: int = 10
    max_cardinality: int = 50


class DriftKsSection(BaseModel):
    pvalue_warn: float = 0.05
    pvalue_alert: float = 0.01
    min_non_null: int = 10


class DriftSection(BaseModel):
    psi: DriftPsiSection = Field(default_factory=DriftPsiSection)
    ks: DriftKsSection = Field(default_factory=DriftKsSection)


class CostSection(BaseModel):
    fp: float = 1.0
    fn: float = 10.0
    grid_size: int = 101


class ExplainabilitySection(BaseModel):
    scoring: str = "average_precision"
    n_repeats: int = 5
    top_k: int = 10
    max_rows: int = 1000


class AppConfig(BaseModel):
    app: AppSection
    paths: PathsSection
    data: DataSection
    model: ModelSection
    training: TrainingSection
    evaluation: EvaluationSection

    # These must be optional in payloads (unit tests omit them),
    # but should still have sensible defaults.
    drift: DriftSection = Field(default_factory=DriftSection)
    cost: CostSection = Field(default_factory=CostSection)
    explainability: ExplainabilitySection = Field(default_factory=ExplainabilitySection)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Config YAML must parse to a mapping/dict.")
        return cls.model_validate(payload)


@dataclass(frozen=True, slots=True)
class ResolvedPaths:
    project_root: Path
    raw_csv: Path
    eval_csv: Path
    artifacts_dir: Path


def _resolve_maybe_relative(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def resolve_paths(cfg: AppConfig, project_root: Path) -> ResolvedPaths:
    """
    Resolves string paths from config into absolute Paths.
    Ensures eval_csv defaults to data/processed/eval.csv when not provided.
    """
    raw_csv = _resolve_maybe_relative(project_root, cfg.paths.raw_csv)
    eval_csv = _resolve_maybe_relative(project_root, cfg.paths.eval_csv)
    artifacts_dir = _resolve_maybe_relative(project_root, cfg.paths.artifacts_dir)

    return ResolvedPaths(
        project_root=project_root.resolve(),
        raw_csv=raw_csv,
        eval_csv=eval_csv,
        artifacts_dir=artifacts_dir,
    )
