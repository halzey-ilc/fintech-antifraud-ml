from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from secrets import token_hex


@dataclass(frozen=True)
class RunId:
    value: str

    @staticmethod
    def new(prefix: str) -> "RunId":
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rnd = token_hex(4)
        safe_prefix = "".join(ch for ch in prefix.lower() if ch.isalnum() or ch in ("-", "_"))
        safe_prefix = safe_prefix or "run"
        return RunId(f"{safe_prefix}-{ts}-{rnd}")
