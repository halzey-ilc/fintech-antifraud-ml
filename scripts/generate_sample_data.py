from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GenSpec:
    rows: int
    seed: int
    start_utc: datetime
    days: int


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_transactions(spec: GenSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)

    base_time = spec.start_utc
    seconds_range = int(spec.days * 24 * 3600)

    event_offset = rng.integers(0, seconds_range, size=spec.rows, endpoint=False)
    event_time = [base_time + timedelta(seconds=int(s)) for s in event_offset]
    event_time_str = [dt.isoformat() for dt in event_time]

    n_customers = max(500, int(spec.rows / 10))
    n_merchants = max(200, int(spec.rows / 20))
    customers = rng.integers(1, n_customers + 1, size=spec.rows)
    merchants = rng.integers(1, n_merchants + 1, size=spec.rows)

    mcc_pool = np.array(["5411", "5812", "5999", "6011", "4111", "4814", "5732", "4900", "4829", "6536"])
    mcc = rng.choice(mcc_pool, size=spec.rows, replace=True)

    country_pool = np.array(["US", "GB", "DE", "FR", "TR", "KZ", "KG", "RU", "AE", "CN"])
    country = rng.choice(country_pool, size=spec.rows, replace=True)

    channel_pool = np.array(["ecomm", "pos", "atm", "inapp"])
    channel = rng.choice(channel_pool, size=spec.rows, replace=True, p=np.array([0.45, 0.45, 0.05, 0.05]))

    currency_pool = np.array(["USD", "EUR", "GBP", "KGS", "KZT", "AED", "CNY", "RUB"])
    currency = rng.choice(currency_pool, size=spec.rows, replace=True)

    is_international = rng.random(spec.rows) < 0.12
    is_card_present = rng.random(spec.rows) < 0.70

    # Amount: log-normal-like (heavy tail)
    amount = np.exp(rng.normal(loc=2.8, scale=0.9, size=spec.rows))
    amount = np.clip(amount, 0.5, 5000.0)

    # Velocity features (counts)
    velocity_1h = rng.poisson(lam=0.8, size=spec.rows)
    velocity_24h = velocity_1h + rng.poisson(lam=2.2, size=spec.rows)

    merchant_risk_score = np.clip(rng.normal(loc=0.35, scale=0.15, size=spec.rows), 0.0, 1.0)
    customer_risk_score = np.clip(rng.normal(loc=0.30, scale=0.18, size=spec.rows), 0.0, 1.0)

    # Fraud probability model (synthetic but plausible):
    # - higher amount increases risk
    # - international + card-not-present increases risk
    # - high velocity increases risk
    # - high merchant/customer risk increases risk
    amt_term = (np.log1p(amount) - 2.5) * 0.9
    intl_term = is_international.astype(float) * 0.8
    cnp_term = (1.0 - is_card_present.astype(float)) * 0.6
    vel_term = (velocity_24h.astype(float) / 10.0) * 0.7
    m_risk_term = (merchant_risk_score - 0.3) * 1.2
    c_risk_term = (customer_risk_score - 0.25) * 1.1

    logits = -3.2 + amt_term + intl_term + cnp_term + vel_term + m_risk_term + c_risk_term
    p = _sigmoid(logits)
    label = (rng.random(spec.rows) < p).astype(int)

    tx_id = [f"tx_{i:08d}" for i in range(spec.rows)]
    device_id = rng.integers(1, max(300, int(spec.rows / 30)) + 1, size=spec.rows)

    df = pd.DataFrame(
        {
            "event_time": event_time_str,
            "transaction_id": tx_id,
            "customer_id": customers.astype(str),
            "merchant_id": merchants.astype(str),
            "device_id": device_id.astype(str),
            "amount": amount.astype(float),
            "currency": currency.astype(str),
            "mcc": mcc.astype(str),
            "country": country.astype(str),
            "channel": channel.astype(str),
            "is_international": is_international.astype(bool),
            "is_card_present": is_card_present.astype(bool),
            "merchant_risk_score": merchant_risk_score.astype(float),
            "customer_risk_score": customer_risk_score.astype(float),
            "velocity_1h": velocity_1h.astype(int),
            "velocity_24h": velocity_24h.astype(int),
            "label": label.astype(int),
        }
    )

    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic transactions dataset for antifraud pipeline.")
    p.add_argument("--out", type=str, required=True, help="Output CSV path")
    p.add_argument("--rows", type=int, default=5000, help="Number of rows to generate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--days", type=int, default=30, help="Time span in days")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spec = GenSpec(
        rows=int(args.rows),
        seed=int(args.seed),
        start_utc=datetime.now(timezone.utc) - timedelta(days=int(args.days)),
        days=int(args.days),
    )

    df = generate_transactions(spec)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Generated: {out_path} rows={len(df)} cols={df.shape[1]}")


if __name__ == "__main__":
    main()
