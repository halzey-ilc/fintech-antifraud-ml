from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunLayout:
    artifacts_dir: Path
    runs_dir: Path
    run_dir: Path
    latest_dir: Path
    latest_run_pointer: Path
    run_id: str


class LocalArtifactStore:
    """
    Local artifacts store with run versioning.

    Layout:
      artifacts/
        runs/<run_id>/*
        latest/*                # copies of key outputs for consumers (API)
        latest_run.json         # pointer to the latest run_id and metadata
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self._artifacts_dir = artifacts_dir.resolve()

    @property
    def artifacts_dir(self) -> Path:
        return self._artifacts_dir

    def prepare_run(self, run_id: str) -> RunLayout:
        runs_dir = (self._artifacts_dir / "runs").resolve()
        run_dir = (runs_dir / run_id).resolve()
        latest_dir = (self._artifacts_dir / "latest").resolve()
        latest_run_pointer = (self._artifacts_dir / "latest_run.json").resolve()

        runs_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        latest_dir.mkdir(parents=True, exist_ok=True)

        return RunLayout(
            artifacts_dir=self._artifacts_dir,
            runs_dir=runs_dir,
            run_dir=run_dir,
            latest_dir=latest_dir,
            latest_run_pointer=latest_run_pointer,
            run_id=run_id,
        )

    def write_json(self, base_dir: Path, rel_path: str, payload: dict[str, Any]) -> Path:
        p = (base_dir / rel_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return p

    def write_text(self, base_dir: Path, rel_path: str, text: str) -> Path:
        p = (base_dir / rel_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def copy_to_latest(self, layout: RunLayout, files: list[Path]) -> None:
        """
        Copy selected run artifacts into artifacts/latest for stable consumers.
        """
        for f in files:
            if not f.exists():
                continue
            dst = (layout.latest_dir / f.name).resolve()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)

    def update_latest_pointer(self, layout: RunLayout, payload: dict[str, Any]) -> Path:
        """
        Update artifacts/latest_run.json with run_id and metadata.
        """
        out = {
            "run_id": layout.run_id,
            **payload,
        }
        return self.write_json(layout.artifacts_dir, "latest_run.json", out)
