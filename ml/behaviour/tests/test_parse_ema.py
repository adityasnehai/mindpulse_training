"""Unit tests for EMA definition/response parsing."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parse_ema import load_ema_definition, parse_ema_response_file, parse_options  # noqa: E402


def test_parse_options_bracket_format():
    options = parse_options("[1]A little stressed, [2]Definitely stressed, [3]Stressed out, ")
    assert options == {1: "A little stressed", 2: "Definitely stressed", 3: "Stressed out"}


def test_parse_options_non_monotonic_stress_scale():
    # Real EMA_definition.json content for Stress/level — codes 4-5 are the
    # *opposite* of codes 1-3, not a continuation of severity.
    options = parse_options(
        "[1]A little stressed, [2]Definitely stressed, [3]Stressed out, [4]Feeling good, [5]Feeling great, "
    )
    assert options[3] == "Stressed out"
    assert options[4] == "Feeling good"
    assert "stress" not in options[4].lower()


def test_parse_options_non_bracket_format_returns_empty():
    # Mood's yes/no questions use "(Yes) 1 2 (No)" — not decodable without
    # inventing a scheme, so this must not be silently guessed.
    assert parse_options("(Yes) 1 2 (No)") == {}


def test_parse_options_empty_string():
    assert parse_options("") == {}


def test_load_ema_definition_structure(tmp_path):
    def_path = tmp_path / "EMA_definition.json"
    def_path.write_text(
        json.dumps(
            [
                {
                    "name": "Stress",
                    "questions": [
                        {"question_id": "level", "question_text": "Right now, I am...", "options": "[1]A little stressed, [2]Feeling great, "},
                        {"question_id": "location", "question_text": "", "options": ""},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    categories = load_ema_definition(def_path)
    assert "Stress" in categories
    assert categories["Stress"]["level"]["options"] == {1: "A little stressed", 2: "Feeling great"}


def test_parse_ema_response_drops_location_and_null_keys(tmp_path):
    path = tmp_path / "Stress_test.json"
    path.write_text(
        json.dumps(
            [
                {"level": "2", "location": "43.7,-72.3", "resp_time": 1364237696},
                {"null": "43.7,-72.3", "resp_time": 1364114401},
            ]
        ),
        encoding="utf-8",
    )
    df = parse_ema_response_file(path, question_ids=["level", "location", "null"])
    assert "location" not in df.columns
    assert "null" not in df.columns
    assert list(df["level"]) == ["2", None]


def test_parse_ema_response_malformed_json_returns_empty(tmp_path):
    path = tmp_path / "Stress_bad.json"
    path.write_text("{not valid", encoding="utf-8")
    df = parse_ema_response_file(path, question_ids=["level"])
    assert df.empty


def test_parse_ema_response_assigns_local_date(tmp_path):
    path = tmp_path / "Stress_test.json"
    # 1364237696 -> 2013-03-25 in America/New_York
    path.write_text(json.dumps([{"level": "1", "resp_time": 1364237696}]), encoding="utf-8")
    df = parse_ema_response_file(path, question_ids=["level"])
    assert df.iloc[0]["date"] == "2013-03-25"
