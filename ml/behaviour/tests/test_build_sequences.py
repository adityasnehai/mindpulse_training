"""Unit tests for 7-day sequence window construction."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_daily_features import ALL_FEATURES  # noqa: E402
from build_sequences import build_windows  # noqa: E402


def make_daily_row(participant_id, date, value=0.5):
    row = {"participant_id": participant_id, "date": date, "coverage": 1.0, "feature_version": "v1"}
    for f in ALL_FEATURES:
        row[f] = value
        row[f"missing_{f}"] = False
    return row


def test_seven_consecutive_days_produces_one_window():
    dates = [f"2013-04-{d:02d}" for d in range(15, 22)]  # 7 consecutive days
    rows = [make_daily_row("u00", d) for d in dates]
    df = pd.DataFrame(rows)

    windows, meta = build_windows(df, window_days=7)
    assert windows.shape == (1, 7, 2 * len(ALL_FEATURES))
    assert meta.iloc[0]["participant_id"] == "u00"
    assert meta.iloc[0]["window_start_date"] == "2013-04-15"
    assert meta.iloc[0]["window_end_date"] == "2013-04-21"


def test_gap_in_dates_prevents_window_across_it():
    # Missing 2013-04-18 entirely -> no 7-day window can include it.
    dates = ["2013-04-15", "2013-04-16", "2013-04-17", "2013-04-19", "2013-04-20", "2013-04-21", "2013-04-22"]
    rows = [make_daily_row("u00", d) for d in dates]
    df = pd.DataFrame(rows)

    windows, meta = build_windows(df, window_days=7)
    assert len(windows) == 0


def test_eight_consecutive_days_produces_two_overlapping_windows():
    dates = [f"2013-04-{d:02d}" for d in range(15, 23)]  # 8 consecutive days
    rows = [make_daily_row("u00", d) for d in dates]
    df = pd.DataFrame(rows)

    windows, meta = build_windows(df, window_days=7)
    assert len(windows) == 2
    assert list(meta["window_start_date"]) == ["2013-04-15", "2013-04-16"]


def test_multiple_participants_are_independent():
    dates_u00 = [f"2013-04-{d:02d}" for d in range(15, 22)]
    dates_u01 = [f"2013-05-{d:02d}" for d in range(1, 8)]
    rows = [make_daily_row("u00", d) for d in dates_u00] + [make_daily_row("u01", d) for d in dates_u01]
    df = pd.DataFrame(rows)

    windows, meta = build_windows(df, window_days=7)
    assert len(windows) == 2
    assert set(meta["participant_id"]) == {"u00", "u01"}


def test_missing_feature_values_are_zero_filled_but_flagged():
    dates = [f"2013-04-{d:02d}" for d in range(15, 22)]
    rows = [make_daily_row("u00", d) for d in dates]
    rows[3][ALL_FEATURES[0]] = np.nan
    rows[3][f"missing_{ALL_FEATURES[0]}"] = True
    df = pd.DataFrame(rows)

    windows, meta = build_windows(df, window_days=7)
    day3_feature0 = windows[0, 3, 0]
    day3_missing0 = windows[0, 3, len(ALL_FEATURES)]
    assert day3_feature0 == 0.0
    assert day3_missing0 == 1.0


def test_empty_input_returns_empty_windows():
    df = pd.DataFrame(columns=["participant_id", "date"] + ALL_FEATURES + [f"missing_{f}" for f in ALL_FEATURES])
    windows, meta = build_windows(df, window_days=7)
    assert windows.shape[0] == 0
    assert meta.empty
