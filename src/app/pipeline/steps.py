from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar


TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class Step(Protocol[TIn, TOut]):
    """Pipeline step contract."""

    def run(self, data: TIn) -> TOut:
        """Run the step and return output."""
        raise NotImplementedError


@dataclass(frozen=True)
class PipelineResult:
    model_path: str
    metrics_path: str
    dataset_fingerprint_path: str
    run_metadata_path: str
