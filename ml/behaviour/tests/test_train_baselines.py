"""Unit tests for the baseline-scoring and evaluation-join logic in train_baselines.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_baselines import evaluate_against_labels, robust_zscore_scores  # noqa: E402


def test_robust_zscore_scores_flags_outlier_higher():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, size=(100, 5))
    normal_point = np.zeros((1, 5))
    outlier_point = np.full((1, 5), 20.0)
    test = np.vstack([normal_point, outlier_point])

    scores = robust_zscore_scores(train, test)
    assert scores[1] > scores[0]


def test_robust_zscore_scores_zero_for_median_point():
    train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    test = np.array([[2.0]])  # exactly the median
    scores = robust_zscore_scores(train, test)
    assert scores[0] == pytest.approx(0.0, abs=1e-6)


def test_evaluate_against_labels_joins_on_participant_and_end_date():
    # 12 matched rows (>= the n_matched>=10 statistical floor in evaluate_against_labels)
    # with a clear monotonic score-vs-severity relationship and both classes present.
    n = 12
    meta = pd.DataFrame(
        {
            "participant_id": ["u00"] * n,
            "window_start_date": [f"2013-04-{i:02d}" for i in range(1, n + 1)],
            "window_end_date": [f"2013-04-{i + 6:02d}" for i in range(1, n + 1)],
        }
    )
    scores = np.arange(n, dtype=float)  # strictly increasing
    severities = np.arange(n) % 5
    labels = pd.DataFrame(
        {
            "participant_id": ["u00"] * n,
            "date": [f"2013-04-{i + 6:02d}" for i in range(1, n + 1)],
            "severity": severities,
            "high_stress": severities >= 3,
        }
    )
    result = evaluate_against_labels(scores, meta, labels)
    assert result["n_matched"] == n
    assert result["spearman_r"] is not None
    assert result["roc_auc"] is not None


def test_evaluate_against_labels_insufficient_matches_returns_none():
    meta = pd.DataFrame(
        {"participant_id": ["u00"], "window_start_date": ["2013-04-01"], "window_end_date": ["2013-04-07"]}
    )
    scores = np.array([1.0])
    labels = pd.DataFrame(columns=["participant_id", "date", "severity", "high_stress"])
    result = evaluate_against_labels(scores, meta, labels)
    assert result["spearman_r"] is None
    assert result["roc_auc"] is None


def test_evaluate_against_labels_single_class_returns_none():
    meta = pd.DataFrame(
        {
            "participant_id": ["u00"] * 12,
            "window_start_date": [f"2013-04-{i:02d}" for i in range(1, 13)],
            "window_end_date": [f"2013-04-{i+6:02d}" for i in range(1, 13)],
        }
    )
    scores = np.arange(12, dtype=float)
    labels = pd.DataFrame(
        {
            "participant_id": ["u00"] * 12,
            "date": [f"2013-04-{i+6:02d}" for i in range(1, 13)],
            "severity": [1] * 12,
            "high_stress": [False] * 12,  # only one class present
        }
    )
    result = evaluate_against_labels(scores, meta, labels)
    assert result["roc_auc"] is None
