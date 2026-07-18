"""Unit tests for LOSO personal-baseline scoring and the index-wiring logic
in run_loso (participant/date filtering, baseline-vs-eval window split,
meta-row alignment) — the exact kind of glue code prone to off-by-index bugs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_loso import BASELINE_DAYS, personal_drift_scores, run_loso  # noqa: E402


def test_personal_drift_scores_matches_spec_formula():
    baseline = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])  # median=(2,2), MAD=(2,2)
    eval_points = np.array([[2.0, 2.0], [10.0, 10.0]])  # exactly at median -> 0; far away -> large
    scores = personal_drift_scores(baseline, eval_points)
    assert scores[0] == pytest.approx(0.0, abs=1e-3)
    assert scores[1] > scores[0]
    # Manually: |10-2|/(2+0.001) = 3.9980... averaged over both dims (identical) = same value
    assert scores[1] == pytest.approx(8 / 2.001, rel=1e-3)


def _make_daily_df(pid, dates):
    return pd.DataFrame({"participant_id": [pid] * len(dates), "date": dates})


def _make_windows_and_meta(pid, dates, window_days=7):
    """Build one window per possible 7-consecutive-day start within `dates`."""
    windows, meta_rows = [], []
    date_set = set(dates)
    for i, start in enumerate(dates):
        expected = [str(pd.Timestamp(start) + pd.Timedelta(days=d)) for d in range(window_days)]
        expected = [pd.Timestamp(e).date().isoformat() for e in expected]
        if all(d in date_set for d in expected):
            windows.append(np.random.default_rng(hash((pid, start)) % 1000).normal(0, 1, size=(7, 24)))
            meta_rows.append({"participant_id": pid, "window_start_date": expected[0], "window_end_date": expected[-1]})
    return windows, meta_rows


def test_participant_with_too_few_valid_days_is_skipped():
    dates = [f"2013-04-{d:02d}" for d in range(1, 10)]  # only 9 valid days, < BASELINE_DAYS=14
    daily_df = _make_daily_df("u00", dates)
    w, m = _make_windows_and_meta("u00", dates)
    windows = np.stack(w) if w else np.empty((0, 7, 24))
    meta = pd.DataFrame(m) if m else pd.DataFrame(columns=["participant_id", "window_start_date", "window_end_date"])

    config = {"training": {"epochs": 1, "batch_size": 4}}
    scores, eval_meta, report = run_loso(windows, meta, daily_df, config)
    assert report["u00"]["status"] == "skipped_insufficient_valid_days"
    assert report["u00"]["valid_days"] == 9


def test_participant_with_enough_days_but_no_post_baseline_windows_is_skipped():
    # Exactly 14 valid days and nothing after -> baseline windows exist, no eval windows.
    dates = [f"2013-04-{d:02d}" for d in range(1, 15)]  # 14 valid days
    daily_df = _make_daily_df("u00", dates)
    w, m = _make_windows_and_meta("u00", dates)
    windows = np.stack(w)
    meta = pd.DataFrame(m)

    config = {"training": {"epochs": 1, "batch_size": 4}}
    scores, eval_meta, report = run_loso(windows, meta, daily_df, config)
    assert report["u00"]["status"] == "skipped_no_baseline_or_eval_windows"
    assert report["u00"]["eval_windows"] == 0


def test_two_participants_baseline_and_eval_split_correctly():
    # u00: 21 consecutive valid days -> baseline = windows ending on/before day 14,
    # eval = windows ending after day 14. u01 provides "other participant" training data.
    dates_00 = [f"2013-04-{d:02d}" for d in range(1, 22)]  # 21 days
    dates_01 = [f"2013-05-{d:02d}" for d in range(1, 22)]  # disjoint dates, same length

    daily_df = pd.concat([_make_daily_df("u00", dates_00), _make_daily_df("u01", dates_01)], ignore_index=True)
    w00, m00 = _make_windows_and_meta("u00", dates_00)
    w01, m01 = _make_windows_and_meta("u01", dates_01)
    windows = np.stack(w00 + w01)
    meta = pd.DataFrame(m00 + m01)

    config = {"training": {"epochs": 1, "batch_size": 4}}
    scores, eval_meta, report = run_loso(windows, meta, daily_df, config)

    assert report["u00"]["status"] == "evaluated"
    assert report["u01"]["status"] == "evaluated"

    u00_baseline_cutoff = dates_00[BASELINE_DAYS - 1]  # 14th valid day
    # Every eval-meta row attributed to u00 must have window_end_date strictly after the cutoff.
    u00_eval_rows = eval_meta[eval_meta["participant_id"] == "u00"]
    assert (u00_eval_rows["window_end_date"] > u00_baseline_cutoff).all()
    assert report["u00"]["baseline_windows"] == 8  # windows with end_date in days 7..14 (indices 0..7)


def test_scores_length_matches_eval_meta_length():
    dates_00 = [f"2013-04-{d:02d}" for d in range(1, 22)]
    dates_01 = [f"2013-05-{d:02d}" for d in range(1, 22)]
    daily_df = pd.concat([_make_daily_df("u00", dates_00), _make_daily_df("u01", dates_01)], ignore_index=True)
    w00, m00 = _make_windows_and_meta("u00", dates_00)
    w01, m01 = _make_windows_and_meta("u01", dates_01)
    windows = np.stack(w00 + w01)
    meta = pd.DataFrame(m00 + m01)

    config = {"training": {"epochs": 1, "batch_size": 4}}
    scores, eval_meta, report = run_loso(windows, meta, daily_df, config)
    assert len(scores) == len(eval_meta)
    assert np.isfinite(scores).all()
