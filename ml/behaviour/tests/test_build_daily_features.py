"""Unit tests for the Milestone 2 daily-feature merge/missingness logic, using
small synthetic per-participant fixture trees (real-shaped file layout)."""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_daily_features import ALL_FEATURES, build_participant_daily_features  # noqa: E402
from timeutils import TZ  # noqa: E402


def ts(y, m, d, h, mi=0):
    return int(datetime(y, m, d, h, mi, tzinfo=TZ).timestamp())


def full_day_samples(y, m, d, activity=0, interval_minutes=2):
    """Dense samples (default every 2 min, well inside the 300s max-gap
    threshold) covering a full day, mirroring real StudentLife sampling
    density rather than sparse hourly points that would all be dropped as
    sensing outages."""
    return [
        (ts(y, m, d, h, mi), activity)
        for h in range(24)
        for mi in range(0, 60, interval_minutes)
    ]


@pytest.fixture
def dataset_root(tmp_path):
    root = tmp_path / "dataset"
    (root / "sensing" / "activity").mkdir(parents=True)
    (root / "sensing" / "phonelock").mkdir(parents=True)
    (root / "sensing" / "phonecharge").mkdir(parents=True)
    return root


def write_activity(root, pid, rows):
    pd.DataFrame(rows, columns=["timestamp", " activity inference"]).to_csv(
        root / "sensing" / "activity" / f"activity_{pid}.csv", index=False
    )


def write_intervals(root, kind, pid, rows):
    pd.DataFrame(rows, columns=["start", "end"]).to_csv(
        root / "sensing" / kind / f"{kind}_{pid}.csv", index=False
    )


def test_day_with_activity_but_no_long_sessions_is_confirmed_zero_not_missing(dataset_root):
    # Full day of stationary samples, no phonelock/phonecharge sessions at all.
    rows = full_day_samples(2013, 4, 15)
    write_activity(dataset_root, "u99", rows)
    write_intervals(dataset_root, "phonelock", "u99", [])
    write_intervals(dataset_root, "phonecharge", "u99", [])

    df = build_participant_daily_features("u99", dataset_root, long_threshold_minutes=60)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["long_lock_total_minutes"] == 0.0
    assert row["missing_long_lock_total_minutes"] == False  # noqa: E712 (confirmed zero, not missing)
    assert row["long_charge_total_minutes"] == 0.0
    assert row["missing_long_charge_total_minutes"] == False  # noqa: E712


def test_day_with_no_activity_data_marks_lock_charge_missing(dataset_root):
    # No activity file coverage for u98 at all -> no rows produced, since
    # lock/charge alone can't establish which day to attribute an "unknown" to
    # without any activity signal in this fixture.
    write_activity(dataset_root, "u98", [])
    write_intervals(dataset_root, "phonelock", "u98", [])
    write_intervals(dataset_root, "phonecharge", "u98", [])
    df = build_participant_daily_features("u98", dataset_root, long_threshold_minutes=60)
    assert df.empty


def test_day_with_partial_activity_gap_marks_lock_missing_when_uncovered(dataset_root):
    # Activity data exists on 2013-04-15 (covered) and no activity at all on
    # 2013-04-16, but a lock session is recorded starting late on the 16th.
    rows = full_day_samples(2013, 4, 15)
    write_activity(dataset_root, "u97", rows)
    lock_rows = [(ts(2013, 4, 16, 22, 0), ts(2013, 4, 16, 23, 30))]  # 90 min, day 16 has no activity coverage
    write_intervals(dataset_root, "phonelock", "u97", lock_rows)
    write_intervals(dataset_root, "phonecharge", "u97", [])

    df = build_participant_daily_features("u97", dataset_root, long_threshold_minutes=60)
    day16 = df[df["date"] == "2013-04-16"].iloc[0]
    # activity_coverage_fraction is NaN/0 on day16 -> lock total is real (session exists)
    # but activity features themselves are missing.
    assert day16["missing_stationary_fraction"] == True  # noqa: E712
    assert day16["long_lock_total_minutes"] == pytest.approx(90.0)


def test_all_expected_columns_present(dataset_root):
    rows = full_day_samples(2013, 4, 15)
    write_activity(dataset_root, "u96", rows)
    write_intervals(dataset_root, "phonelock", "u96", [])
    write_intervals(dataset_root, "phonecharge", "u96", [])
    df = build_participant_daily_features("u96", dataset_root, long_threshold_minutes=60)
    for feature in ALL_FEATURES:
        assert feature in df.columns
        assert f"missing_{feature}" in df.columns
    assert "coverage" in df.columns
    assert "feature_version" in df.columns
    assert "participant_id" in df.columns


def test_no_files_returns_empty(dataset_root):
    df = build_participant_daily_features("u95", dataset_root, long_threshold_minutes=60)
    assert df.empty
