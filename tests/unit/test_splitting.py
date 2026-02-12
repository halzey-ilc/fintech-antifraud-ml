from __future__ import annotations

import pandas as pd

from app.data.splitting import SplitSpec, split_dataset


def test_time_group_split_has_no_group_overlap() -> None:
    df = pd.DataFrame(
        {
            "event_time": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-02-01T00:00:00Z",
                "2024-02-02T00:00:00Z",
                "2024-03-01T00:00:00Z",
                "2024-03-02T00:00:00Z",
            ],
            "customer_id": ["A", "A", "B", "B", "C", "C"],
            "amount": [10, 20, 30, 40, 50, 60],
            "mcc": ["x", "x", "y", "y", "z", "z"],
            "label": [0, 1, 0, 0, 1, 0],
        }
    )

    spec = SplitSpec(
        test_size=0.34,
        random_seed=42,
        strategy="time",
        label_column="label",
        time_column="event_time",
        group_column="customer_id",
    )
    res = split_dataset(df, spec)

    train_groups = set(res.X_train["customer_id"].tolist())
    test_groups = set(res.X_test["customer_id"].tolist())
    assert train_groups.isdisjoint(test_groups)


def test_random_group_split_has_no_group_overlap() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "amount": [1, 2, 3, 4, 5, 6, 7, 8],
            "label": [0, 1, 0, 0, 1, 0, 0, 1],
        }
    )

    spec = SplitSpec(
        test_size=0.25,
        random_seed=1,
        strategy="random",
        label_column="label",
        group_column="customer_id",
    )
    res = split_dataset(df, spec)

    train_groups = set(res.X_train["customer_id"].tolist())
    test_groups = set(res.X_test["customer_id"].tolist())
    assert train_groups.isdisjoint(test_groups)
