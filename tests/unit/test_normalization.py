from __future__ import annotations

from datetime import datetime

from app.api.normalization import normalize_transaction_record


def test_normalize_transaction_record_basic() -> None:
    dt = datetime(2026, 1, 1, 12, 0, 0)
    rec = {
        "amount": 10.0,
        "country": " us ",
        "currency": "usd",
        "mcc": " 54-11 ",
        "channel": " ECOMM ",
        "merchant_id": "  m123 ",
        "event_time": dt,
        "extra_field": "  x  ",
    }
    out = normalize_transaction_record(rec)

    assert out["country"] == "US"
    assert out["currency"] == "USD"
    assert out["mcc"] == "5411"
    assert out["channel"] == "ecomm"
    assert out["merchant_id"] == "m123"
    assert out["extra_field"] == "x"
    assert out["event_time"] == dt


def test_normalize_transaction_record_event_time_invalid_type_becomes_none() -> None:
    rec = {"amount": 1.0, "event_time": "2026-01-01T00:00:00Z"}
    out = normalize_transaction_record(rec)
    assert out["event_time"] is None
