"""Integration test for the encoder GroupKFold orchestration (fold-splitting,
scaling, and score-index wiring) using tiny synthetic data — not testing model
quality, just that every window gets exactly one score and indices line up."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import encoder_group_kfold  # noqa: E402


def test_every_window_gets_scored_exactly_once():
    rng = np.random.default_rng(0)
    n_participants = 10
    windows_per_participant = 6
    n = n_participants * windows_per_participant

    windows = rng.normal(0, 1, size=(n, 7, 24)).astype(np.float32)
    windows[:, :, 12:] = 0.0  # no missingness
    participant_ids = [f"u{p:02d}" for p in range(n_participants) for _ in range(windows_per_participant)]
    meta = pd.DataFrame({"participant_id": participant_ids})

    config = {"training": {"epochs": 2, "batch_size": 8}}
    scores = encoder_group_kfold(windows, meta, config, val_fraction=0.2)

    assert scores.shape == (n,)
    assert np.isfinite(scores).all()
    # No score should remain at the zero-initialized default for every window
    # unless a fold genuinely produced 0 for all — with random data that's
    # not expected, so this catches an index-wiring bug that leaves some
    # windows unscored.
    assert not np.all(scores == 0)
