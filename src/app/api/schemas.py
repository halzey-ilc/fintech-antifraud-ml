from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from pydantic.functional_validators import field_validator


class ScoreRecord(BaseModel):
    # Allow extra keys (feature columns) in the request payload.
    model_config = ConfigDict(extra="allow")

    amount: float = Field(..., ge=0)
    velocity_24h: float = Field(..., ge=0)

    # Optional fields (validated after normalization).
    mcc: str | None = Field(default=None, pattern=r"^\d{4}$")
    country: str | None = Field(default=None, min_length=2, max_length=2, pattern=r"^[A-Z]{2}$")
    channel: str | None = None
    is_international: bool | None = None

    @field_validator("country", mode="before")
    @classmethod
    def _normalize_country(cls, v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip().upper()
        return s

    @field_validator("mcc", mode="before")
    @classmethod
    def _normalize_mcc(cls, v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip()
        digits = re.sub(r"\D+", "", s)
        return digits

    @field_validator("channel", mode="before")
    @classmethod
    def _normalize_channel(cls, v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip().upper()
        return s


class ScoreRequest(BaseModel):
    records: list[ScoreRecord] = Field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        # Convert validated & normalized records into a DataFrame.
        rows: list[dict[str, Any]] = [r.model_dump() for r in self.records]
        return pd.DataFrame(rows)


class ScoreItem(BaseModel):
    # Required by integration tests
    proba_fraud: float

    # Convenience/compat fields (tests may also check them)
    score: float
    is_fraud: bool
    decision: str


class ScoreResponse(BaseModel):
    items: list[ScoreItem] = Field(default_factory=list)
    threshold: float | None = None
    model_loaded: bool
    error: str | None = None
