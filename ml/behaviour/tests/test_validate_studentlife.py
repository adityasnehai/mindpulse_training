"""Unit tests for the Milestone 1 validator's parsing logic.

These use small, explicitly-labelled synthetic fixtures to test the *code paths*
(CSV/JSON parsing, participant-id extraction, missing-file handling). They are
not a substitute for running validate_studentlife.py against the real archive —
no test here claims to represent actual StudentLife data.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from validate_studentlife import (  # noqa: E402
    inventory_activity,
    inventory_ema_definition,
    inventory_ema_responses,
    inventory_intervals,
    inventory_user_info,
)


def test_inventory_activity_missing_dir(tmp_path):
    result = inventory_activity(tmp_path / "does_not_exist")
    assert result == {"present": False, "participants": {}}


def test_inventory_activity_parses_participant_id_and_span(tmp_path):
    activity_dir = tmp_path / "activity"
    activity_dir.mkdir()
    df = pd.DataFrame({"timestamp": [1364440000, 1364440500, 1364441000], "activity_inference": [0, 1, 0]})
    df.to_csv(activity_dir / "activity_u00.csv", index=False)

    result = inventory_activity(activity_dir)
    assert result["present"] is True
    assert "u00" in result["participants"]
    assert result["participants"]["u00"]["rows"] == 3
    assert result["participants"]["u00"]["min_timestamp"] == 1364440000
    assert result["participants"]["u00"]["max_timestamp"] == 1364441000


def test_inventory_intervals_missing_dir(tmp_path):
    result = inventory_intervals(tmp_path / "does_not_exist", "phonelock")
    assert result == {"present": False, "participants": {}}


def test_inventory_intervals_parses_rows(tmp_path):
    lock_dir = tmp_path / "phonelock"
    lock_dir.mkdir()
    df = pd.DataFrame({"start": [1000, 5000], "end": [4600, 9000]})
    df.to_csv(lock_dir / "phonelock_u01.csv", index=False)

    result = inventory_intervals(lock_dir, "phonelock")
    assert result["participants"]["u01"]["rows"] == 2


def test_inventory_ema_definition_missing(tmp_path):
    result = inventory_ema_definition(tmp_path / "EMA_definition.json")
    assert result == {"present": False}


def test_inventory_ema_definition_reads_categories(tmp_path):
    def_path = tmp_path / "EMA_definition.json"
    def_path.write_text(
        json.dumps(
            [
                {"name": "Stress", "questions": [{"question_id": "level", "question_text": "..."}]},
                {"name": "Mood", "questions": [{"question_id": "level", "question_text": "..."}]},
            ]
        ),
        encoding="utf-8",
    )

    result = inventory_ema_definition(def_path)
    assert result["present"] is True
    assert result["category_count"] == 2
    assert set(result["categories"].keys()) == {"Stress", "Mood"}


def test_inventory_ema_responses_counts_entries(tmp_path):
    response_dir = tmp_path / "Stress"
    response_dir.mkdir()
    (response_dir / "Stress_u02.json").write_text(
        json.dumps([{"resp_time": "1", "level": "2"}, {"resp_time": "2", "level": "3"}]),
        encoding="utf-8",
    )

    result = inventory_ema_responses(response_dir)
    assert result["participants"]["u02"]["response_count"] == 2


def test_inventory_ema_responses_handles_malformed_json(tmp_path):
    response_dir = tmp_path / "Mood"
    response_dir.mkdir()
    (response_dir / "Mood_u03.json").write_text("{not valid json", encoding="utf-8")

    result = inventory_ema_responses(response_dir)
    assert result["participants"]["u03"]["parse_error"] is True


def test_inventory_user_info_missing(tmp_path):
    result = inventory_user_info(tmp_path / "user_info.csv")
    assert result == {"present": False}


def test_inventory_user_info_reads_rows_and_columns(tmp_path):
    path = tmp_path / "user_info.csv"
    pd.DataFrame({"uid": ["u00", "u01"], "gender": ["m", "f"]}).to_csv(path, index=False)

    result = inventory_user_info(path)
    assert result["present"] is True
    assert result["rows"] == 2
    assert result["columns"] == ["uid", "gender"]
