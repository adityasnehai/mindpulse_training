"""Unit tests for the shared keyword-level safety exclusion filter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safety_filters import contains_excluded_content  # noqa: E402


def test_flags_self_harm_language():
    assert contains_excluded_content("I've been thinking about hurting myself, it's a form of self-harm")


def test_flags_suicide_language():
    assert contains_excluded_content("Sometimes I think about suicide")


def test_flags_medication_language():
    assert contains_excluded_content("My doctor prescribed a new antidepressant")


def test_flags_diagnosis_language():
    assert contains_excluded_content("I was diagnosed with depression last year")


def test_flags_alcohol_abuse_language():
    assert contains_excluded_content("I have a drinking problem and I'm an alcoholic")


def test_flags_serious_illness_language():
    assert contains_excluded_content("I'm sure your friend was really frightened at the prospect of cancer.")


def test_flags_death_language():
    assert contains_excluded_content("My grandmother died last year and it was very hard.")


def test_does_not_flag_ordinary_text():
    assert not contains_excluded_content("I had a really tiring day at work and I'm a little annoyed.")


def test_does_not_flag_empty_string():
    assert not contains_excluded_content("")


def test_case_insensitive():
    assert contains_excluded_content("SUICIDE hotline number")
