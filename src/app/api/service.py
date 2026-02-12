from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joblib import load

from app.core.logging import get_logger

log = get_logger("api")


@dataclass(frozen=True, slots=True)
class ModelLoadResult:
    model: Any | None
    model_path: Path | None
    selection: dict[str, Any] | None
    threshold: float | None
    error: str | None


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        log.warning("selection.read.failed path=%s error=%s", str(path), str(exc))
        return None


def _extract_threshold(selection: dict[str, Any] | None) -> float | None:
    if not selection:
        return None

    try:
        selected = selection.get("selected") or {}
        kind = selected.get("kind")
        per_model = selection.get("per_model") or {}
        if kind and kind in per_model:
            best = ((per_model[kind] or {}).get("cost_report") or {}).get("best") or {}
            thr = best.get("threshold")
            if thr is not None:
                return float(thr)
    except Exception:  # noqa: BLE001
        return None

    # Fallbacks (if format differs)
    for key_path in (
        ("threshold",),
        ("cost_report", "best", "threshold"),
    ):
        cur: Any = selection
        ok = True
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and cur is not None:
            try:
                return float(cur)
            except Exception:  # noqa: BLE001
                return None

    return None


def load_model_from_artifacts(artifacts_dir: Path) -> ModelLoadResult:
    """
    Load model and optional selection metadata from artifacts directory.

    Expected files:
      - model.joblib
      - model_selection.json (optional)
    """
    try:
        model_path = artifacts_dir / "model.joblib"
        selection_path = artifacts_dir / "model_selection.json"

        selection = _safe_read_json(selection_path)
        threshold = _extract_threshold(selection)

        model = load(model_path)
        return ModelLoadResult(
            model=model,
            model_path=model_path,
            selection=selection,
            threshold=threshold,
            error=None,
        )
    except FileNotFoundError as exc:
        return ModelLoadResult(
            model=None,
            model_path=None,
            selection=None,
            threshold=None,
            error=f"Artifacts not found: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return ModelLoadResult(
            model=None,
            model_path=None,
            selection=None,
            threshold=None,
            error=f"Failed to load model: {exc}",
        )
