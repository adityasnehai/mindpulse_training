"""Unit tests for stress-severity remapping and within-person high-stress labels."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ema_labels import build_high_stress_labels, stress_severity_map  # noqa: E402


REAL_STRESS_DEFINITION = [
    {
        "name": "Stress",
        "questions": [
            {
                "question_id": "level",
                "question_text": "Right now, I am...",
                "options": "[1]A little stressed, [2]Definitely stressed, [3]Stressed out, [4]Feeling good, [5]Feeling great, ",
            }
        ],
    }
]


@pytest.fixture
def ema_definition_path(tmp_path):
    path = tmp_path / "EMA_definition.json"
    path.write_text(json.dumps(REAL_STRESS_DEFINITION), encoding="utf-8")
    return path


def test_severity_map_orders_stressed_out_highest(ema_definition_path):
    severity_map = stress_severity_map(ema_definition_path)
    assert severity_map[3] > severity_map[2] > severity_map[1]  # stressed-out > definitely > a little


def test_severity_map_feeling_good_is_lower_than_a_little_stressed(ema_definition_path):
    # This is the crux of the whole module: code 4 ("Feeling good") must NOT
    # be treated as more severe than code 1 ("A little stressed") just
    # because 4 > 1 numerically.
    severity_map = stress_severity_map(ema_definition_path)
    assert severity_map[4] < severity_map[1]
    assert severity_map[5] < severity_map[4]


def test_severity_map_raises_on_unrecognized_label(tmp_path):
    bad_def = [
        {
            "name": "Stress",
            "questions": [{"question_id": "level", "question_text": "...", "options": "[1]Some new option text, "}],
        }
    ]
    path = tmp_path / "EMA_definition.json"
    path.write_text(json.dumps(bad_def), encoding="utf-8")
    with pytest.raises(ValueError):
        stress_severity_map(path)


def test_build_high_stress_labels_within_person_quartile(tmp_path, ema_definition_path):
    response_dir = tmp_path / "Stress"
    response_dir.mkdir()
    # 8 responses across 8 days; severities via labels: mostly low, with two clearly high days.
    responses = [
        {"level": "4", "resp_time": 1364100000 + i * 86400}  # feeling good (low severity) for days 0-5
        for i in range(6)
    ] + [
        {"level": "3", "resp_time": 1364100000 + 6 * 86400},  # stressed out (high) day 6
        {"level": "3", "resp_time": 1364100000 + 7 * 86400},  # stressed out (high) day 7
    ]
    (response_dir / "Stress_u00.json").write_text(json.dumps(responses), encoding="utf-8")

    labels = build_high_stress_labels(response_dir, ema_definition_path, ["u00"])
    assert len(labels) == 8
    high_days = labels[labels["high_stress"]]
    assert len(high_days) >= 1
    # raw level "3" decodes to option text "Stressed out" -> mapped severity 4 (highest).
    # raw level "4" decodes to option text "Feeling good" -> mapped severity 1 (low).
    assert all(labels.loc[labels["severity"] == 4, "high_stress"])
    assert all(labels.loc[labels["severity"] == 1, "high_stress"] == False)  # noqa: E712


def test_build_high_stress_labels_skips_participants_with_too_few_responses(tmp_path, ema_definition_path):
    response_dir = tmp_path / "Stress"
    response_dir.mkdir()
    (response_dir / "Stress_u01.json").write_text(
        json.dumps([{"level": "3", "resp_time": 1364100000}]), encoding="utf-8"
    )
    labels = build_high_stress_labels(response_dir, ema_definition_path, ["u01"])
    assert labels.empty


def test_build_high_stress_labels_missing_participant_file_skipped(tmp_path, ema_definition_path):
    response_dir = tmp_path / "Stress"
    response_dir.mkdir()
    labels = build_high_stress_labels(response_dir, ema_definition_path, ["u_nonexistent"])
    assert labels.empty
