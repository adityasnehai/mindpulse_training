"""Unit tests for timezone-aware day-boundary and cross-midnight splitting logic."""

import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timeutils import TZ, day_bounds, local_date, night_overlap_seconds, split_interval_by_day  # noqa: E402


def ts(y, m, d, h=0, mi=0, s=0):
    return int(datetime(y, m, d, h, mi, s, tzinfo=TZ).timestamp())


def test_local_date_basic():
    assert local_date(ts(2013, 4, 15, 10, 30)) == "2013-04-15"


def test_day_bounds_span_full_day():
    start, end = day_bounds("2013-04-15")
    assert start.hour == 0 and start.minute == 0 and start.second == 0
    assert end.hour == 23 and end.minute == 59 and end.second == 59


def test_split_interval_within_single_day():
    start = ts(2013, 4, 15, 10, 0)
    end = ts(2013, 4, 15, 11, 0)
    pieces = split_interval_by_day(start, end)
    assert len(pieces) == 1
    assert pieces[0][0] == "2013-04-15"
    assert pieces[0][1] == start
    assert pieces[0][2] == end


def test_split_interval_crosses_one_midnight():
    start = ts(2013, 4, 15, 23, 40)
    end = ts(2013, 4, 16, 1, 15)
    pieces = split_interval_by_day(start, end)
    assert len(pieces) == 2
    assert pieces[0][0] == "2013-04-15"
    assert pieces[1][0] == "2013-04-16"
    # First piece ends exactly at midnight, second starts exactly at midnight.
    assert pieces[0][2] == pieces[1][1]
    midnight = ts(2013, 4, 16, 0, 0)
    assert pieces[0][2] == midnight


def test_split_interval_crosses_two_midnights():
    start = ts(2013, 4, 15, 23, 0)
    end = ts(2013, 4, 17, 1, 0)
    pieces = split_interval_by_day(start, end)
    dates = [p[0] for p in pieces]
    assert dates == ["2013-04-15", "2013-04-16", "2013-04-17"]
    total_duration = sum(p[2] - p[1] for p in pieces)
    assert total_duration == end - start


def test_split_interval_empty_when_end_before_start():
    assert split_interval_by_day(ts(2013, 4, 15, 10, 0), ts(2013, 4, 15, 9, 0)) == []


def test_split_interval_handles_dst_spring_forward():
    # US DST spring-forward 2013: clocks jump 2:00am -> 3:00am on March 10.
    start = ts(2013, 3, 10, 1, 0)
    end = ts(2013, 3, 11, 1, 0)
    pieces = split_interval_by_day(start, end)
    dates = [p[0] for p in pieces]
    assert dates == ["2013-03-10", "2013-03-11"]
    # Wall-clock duration is still correctly apportioned around the missing hour.
    total_duration = sum(p[2] - p[1] for p in pieces)
    assert total_duration == end - start


def test_night_overlap_fully_inside_night_window():
    date_str = "2013-04-15"
    start = ts(2013, 4, 15, 1, 0)
    end = ts(2013, 4, 15, 3, 0)
    assert night_overlap_seconds(start, end, date_str) == 2 * 3600


def test_night_overlap_partial():
    date_str = "2013-04-15"
    start = ts(2013, 4, 15, 5, 0)
    end = ts(2013, 4, 15, 8, 0)
    # Only 05:00-06:00 (1 hour) is inside the 00:00-06:00 night window.
    assert night_overlap_seconds(start, end, date_str) == 3600


def test_night_overlap_none_during_day():
    date_str = "2013-04-15"
    start = ts(2013, 4, 15, 12, 0)
    end = ts(2013, 4, 15, 14, 0)
    assert night_overlap_seconds(start, end, date_str) == 0
