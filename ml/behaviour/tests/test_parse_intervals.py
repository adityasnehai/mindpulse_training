"""Unit tests for phonelock/phonecharge interval parsing and daily aggregation."""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parse_intervals import daily_interval_features, merge_overlapping_intervals, parse_interval_file  # noqa: E402
from timeutils import TZ  # noqa: E402


def ts(y, m, d, h, mi=0, s=0):
    return int(datetime(y, m, d, h, mi, s, tzinfo=TZ).timestamp())


def test_merge_overlapping_intervals_combines_overlaps():
    # Real-data shape: three intervals for u13 on 2013-04-28 that overlap
    # each other (start of one falls before the end of a previous one).
    intervals = [(100, 300), (250, 500), (450, 600)]
    merged = merge_overlapping_intervals(intervals)
    assert merged == [(100, 600)]


def test_merge_overlapping_intervals_keeps_disjoint_separate():
    intervals = [(100, 200), (500, 600)]
    assert merge_overlapping_intervals(intervals) == [(100, 200), (500, 600)]


def test_merge_overlapping_intervals_merges_touching():
    intervals = [(100, 200), (200, 300)]
    assert merge_overlapping_intervals(intervals) == [(100, 300)]


def test_merge_overlapping_intervals_unsorted_input():
    intervals = [(500, 600), (100, 300), (250, 550)]
    assert merge_overlapping_intervals(intervals) == [(100, 600)]


def test_merge_overlapping_intervals_empty():
    assert merge_overlapping_intervals([]) == []


def test_overlapping_raw_rows_never_exceed_24_hours_per_day(interval_csv):
    # Reproduces the real bug found in u13/2013-04-28: three overlapping
    # phonelock rows on the same day must not sum to more than 1440 minutes.
    rows = [
        (ts(2013, 4, 28, 8, 0), ts(2013, 4, 28, 9, 0)),
        (ts(2013, 4, 28, 8, 30), ts(2013, 4, 28, 10, 0)),
        (ts(2013, 4, 28, 9, 45), ts(2013, 4, 28, 11, 0)),
    ]
    segments = parse_interval_file(interval_csv(rows))
    daily = daily_interval_features(segments, prefix="lock", long_threshold_minutes=0)
    assert daily.iloc[0]["long_lock_total_minutes"] <= 1440
    # 08:00 -> 11:00 merged = exactly 180 minutes, not 90+90+75=255.
    assert daily.iloc[0]["long_lock_total_minutes"] == pytest.approx(180.0)


@pytest.fixture
def interval_csv(tmp_path):
    def _make(rows):
        path = tmp_path / "intervals_test.csv"
        pd.DataFrame(rows, columns=["start", "end"]).to_csv(path, index=False)
        return path

    return _make


def test_single_day_long_session_counted(interval_csv):
    rows = [(ts(2013, 4, 15, 22, 0), ts(2013, 4, 16, 0, 30))]  # 2.5h, but crosses midnight
    segments = parse_interval_file(interval_csv(rows))
    daily = daily_interval_features(segments, prefix="lock", long_threshold_minutes=60)
    # session touches two days; both get counted since original session >= 60 min
    assert set(daily["date"]) == {"2013-04-15", "2013-04-16"}
    day1 = daily[daily["date"] == "2013-04-15"].iloc[0]
    day2 = daily[daily["date"] == "2013-04-16"].iloc[0]
    assert day1["long_lock_total_minutes"] == pytest.approx(120.0)
    assert day2["long_lock_total_minutes"] == pytest.approx(30.0)
    assert day1["long_lock_session_count"] == 1
    assert day2["long_lock_session_count"] == 1


def test_short_session_below_threshold_excluded(interval_csv):
    rows = [(ts(2013, 4, 15, 10, 0), ts(2013, 4, 15, 10, 30))]  # 30 min, below 60-min threshold
    segments = parse_interval_file(interval_csv(rows))
    daily = daily_interval_features(segments, prefix="lock", long_threshold_minutes=60)
    assert daily.empty


def test_session_split_by_midnight_still_eligible_by_original_total(interval_csv):
    # 40 min before midnight + 40 min after = 80 min original session, both pieces
    # individually under 60 min but the session as a whole clears the threshold.
    rows = [(ts(2013, 4, 15, 23, 20), ts(2013, 4, 16, 0, 40))]
    segments = parse_interval_file(interval_csv(rows))
    daily = daily_interval_features(segments, prefix="lock", long_threshold_minutes=60)
    assert len(daily) == 2
    assert daily["long_lock_total_minutes"].sum() == pytest.approx(80.0)


def test_longest_and_count_with_multiple_sessions_same_day(interval_csv):
    rows = [
        (ts(2013, 4, 15, 1, 0), ts(2013, 4, 15, 2, 0)),   # 60 min
        (ts(2013, 4, 15, 10, 0), ts(2013, 4, 15, 11, 30)),  # 90 min
    ]
    segments = parse_interval_file(interval_csv(rows))
    daily = daily_interval_features(segments, prefix="lock", long_threshold_minutes=60)
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["long_lock_session_count"] == 2
    assert row["longest_long_lock_minutes"] == pytest.approx(90.0)
    assert row["long_lock_total_minutes"] == pytest.approx(150.0)


def test_night_overlap_computed_for_charging(interval_csv):
    rows = [(ts(2013, 4, 15, 4, 0), ts(2013, 4, 15, 7, 0))]  # 3h, all inside/crossing night window
    segments = parse_interval_file(interval_csv(rows))
    daily = daily_interval_features(
        segments, prefix="charge", long_threshold_minutes=60, include_night_overlap=True
    )
    row = daily.iloc[0]
    # 04:00-06:00 is inside the night window (2h), 06:00-07:00 is not.
    assert row["night_charge_overlap_minutes"] == pytest.approx(120.0)


def test_empty_input(interval_csv):
    segments = parse_interval_file(interval_csv([]))
    daily = daily_interval_features(segments, prefix="lock")
    assert daily.empty
