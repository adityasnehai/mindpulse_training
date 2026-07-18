"""Unit tests for activity parsing/aggregation, using small synthetic fixtures
that mirror the real file's exact column names (including the leading-space
quirk confirmed against the actual archive)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parse_activity import daily_activity_fractions, parse_activity_file  # noqa: E402


@pytest.fixture
def activity_csv(tmp_path):
    def _make(rows):
        path = tmp_path / "activity_test.csv"
        df = pd.DataFrame(rows, columns=["timestamp", " activity inference"])
        df.to_csv(path, index=False)
        return path

    return _make


def ts(y, m, d, h, mi=0, s=0):
    from datetime import datetime
    from timeutils import TZ

    return int(datetime(y, m, d, h, mi, s, tzinfo=TZ).timestamp())


def test_parses_leading_space_column_and_labels(activity_csv):
    rows = [
        (ts(2013, 4, 15, 10, 0), 0),
        (ts(2013, 4, 15, 10, 1), 1),
        (ts(2013, 4, 15, 10, 2), 2),
        (ts(2013, 4, 15, 10, 3), 3),
    ]
    segments = parse_activity_file(activity_csv(rows))
    assert set(segments["label"]) == {"stationary", "walking", "running"}
    assert len(segments) == 3  # last sample has no following sample -> no segment


def test_large_gap_is_excluded_not_attributed(activity_csv):
    rows = [
        (ts(2013, 4, 15, 10, 0), 0),
        (ts(2013, 4, 15, 22, 0), 0),  # 12-hour gap, far beyond max_gap_seconds
    ]
    segments = parse_activity_file(activity_csv(rows), max_gap_seconds=300)
    assert segments.empty


def test_small_gap_is_attributed(activity_csv):
    rows = [
        (ts(2013, 4, 15, 10, 0, 0), 1),
        (ts(2013, 4, 15, 10, 0, 30), 1),
    ]
    segments = parse_activity_file(activity_csv(rows), max_gap_seconds=300)
    assert len(segments) == 1
    assert segments.iloc[0]["duration_seconds"] == 30


def test_segment_crossing_midnight_is_split(activity_csv):
    rows = [
        (ts(2013, 4, 15, 23, 55), 1),
        (ts(2013, 4, 16, 0, 5), 1),
    ]
    segments = parse_activity_file(activity_csv(rows), max_gap_seconds=3600)
    assert set(segments["date"]) == {"2013-04-15", "2013-04-16"}


def test_daily_fractions_sum_to_one_when_fully_covered(activity_csv):
    rows = [
        (ts(2013, 4, 15, 10, 0, 0), 0),
        (ts(2013, 4, 15, 10, 0, 30), 1),
        (ts(2013, 4, 15, 10, 1, 0), 2),
        (ts(2013, 4, 15, 10, 1, 30), 0),
    ]
    segments = parse_activity_file(activity_csv(rows), max_gap_seconds=60)
    daily = daily_activity_fractions(segments)
    assert len(daily) == 1
    total = (
        daily.iloc[0]["stationary_fraction"]
        + daily.iloc[0]["walking_fraction"]
        + daily.iloc[0]["running_fraction"]
        + daily.iloc[0]["unknown_activity_fraction"]
    )
    assert total == pytest.approx(1.0)


def test_daily_fractions_empty_input(activity_csv):
    segments = parse_activity_file(activity_csv([]))
    daily = daily_activity_fractions(segments)
    assert daily.empty
