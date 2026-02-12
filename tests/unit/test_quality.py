from __future__ import annotations

import pandas as pd

from app.data.quality import QualityReportSpec, build_quality_report


def test_quality_report_handles_bool_columns() -> None:
    df = pd.DataFrame(
        {
            "event_time": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            "amount": [10.0, 20.0],
            "is_international": [True, False],
            "label": [0, 1],
        }
    )

    report = build_quality_report(df, QualityReportSpec(label_column="label"))

    assert report["shape"]["rows"] == 2
    assert "amount" in report["numeric"]
    # bool must not be treated as numeric (no quantile)
    assert "is_international" not in report["numeric"]
    assert "is_international" in report["categorical"]
