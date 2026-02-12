from __future__ import annotations

from datetime import datetime
from typing import Any


def _norm_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _norm_upper(v: Any) -> str | None:
    s = _norm_str(v)
    return None if s is None else s.upper()


def _norm_lower(v: Any) -> str | None:
    s = _norm_str(v)
    return None if s is None else s.lower()


def _digits_only(v: Any) -> str | None:
    s = _norm_str(v)
    if s is None:
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if digits else None


def normalize_transaction_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize inbound transaction record to stabilize the API contract.

    Rules:
    - Trim string fields
    - country, currency -> uppercase
    - channel -> lowercase
    - mcc -> digits only
    - keep unknown extra fields (trim strings where feasible)
    """
    out: dict[str, Any] = {}

    for k, v in record.items():
        if isinstance(v, str):
            out[k] = _norm_str(v)
        else:
            out[k] = v

    if "country" in out:
        out["country"] = _norm_upper(out.get("country"))
    if "currency" in out:
        out["currency"] = _norm_upper(out.get("currency"))
    if "channel" in out:
        out["channel"] = _norm_lower(out.get("channel"))
    if "mcc" in out:
        out["mcc"] = _digits_only(out.get("mcc"))

    # Defensive: ensure event_time stays datetime or None, do not coerce strings here (pydantic already did).
    if "event_time" in out and out["event_time"] is not None and not isinstance(out["event_time"], datetime):
        out["event_time"] = None

    return out
