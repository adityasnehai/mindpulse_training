"""Unit tests for JSON extraction from raw model output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_llm import extract_json_object  # noqa: E402


def test_extract_pure_json():
    text = '{"acknowledgement": "a", "question": "b", "action": {"id": "x"}}'
    result = extract_json_object(text)
    assert result == {"acknowledgement": "a", "question": "b", "action": {"id": "x"}}


def test_extract_json_with_surrounding_whitespace():
    text = '  \n{"a": 1}\n  '
    assert extract_json_object(text) == {"a": 1}


def test_extract_returns_none_for_non_json():
    assert extract_json_object("This is not JSON at all.") is None


def test_extract_returns_none_for_malformed_json():
    assert extract_json_object("{not: valid, json}") is None


def test_extract_json_ignores_leading_trailing_text():
    text = 'Here is the answer: {"a": 1} Hope that helps!'
    assert extract_json_object(text) == {"a": 1}
