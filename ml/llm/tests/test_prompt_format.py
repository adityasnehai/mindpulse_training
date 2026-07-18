"""Unit tests for prompt construction and loss masking, using the REAL Gemma
3 tokenizer (already downloaded and cached) — not a mock, since the exact
token boundary between prompt and target depends on the real chat template
and tokenizer behavior, which must not be assumed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompt_format import build_messages, build_model_turn, build_user_turn, tokenize_and_mask  # noqa: E402

SAMPLE_RECORD = {
    "input": {
        "goal": "stress_management",
        "changes": [{"feature": "stationary_fraction", "direction": "increase", "magnitude": "moderate", "duration_days": 4}],
        "user_context": "work_or_study_pressure",
        "available_time_minutes": 2,
        "previous_helpful_actions": [],
        "previous_unhelpful_actions": [],
        "candidate_actions": ["two_minute_shutdown"],
    },
    "output": {
        "acknowledgement": "Your routine has shifted.",
        "question": "Would one small step help?",
        "action": {"id": "two_minute_shutdown", "title": "Two minute shutdown", "duration_minutes": 2, "steps": ["Step one.", "Step two."]},
    },
}


@pytest.fixture(scope="module")
def real_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/gemma-3-270m-it")


def test_build_user_turn_contains_json_input():
    turn = build_user_turn(SAMPLE_RECORD["input"])
    assert "stress_management" in turn
    assert "two_minute_shutdown" in turn


def test_build_model_turn_is_valid_json_only():
    import json

    turn = build_model_turn(SAMPLE_RECORD["output"])
    parsed = json.loads(turn)  # must not raise
    assert parsed["action"]["id"] == "two_minute_shutdown"


def test_build_messages_has_user_then_assistant_roles():
    messages = build_messages(SAMPLE_RECORD)
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_tokenize_and_mask_prompt_tokens_are_masked(real_tokenizer):
    result = tokenize_and_mask(SAMPLE_RECORD, real_tokenizer, max_length=512)
    labels = result["labels"]
    input_ids = result["input_ids"]

    assert len(labels) == len(input_ids)
    # At least the first several tokens (the whole user turn) must be masked.
    assert labels[0] == -100
    # Some tokens near the end (the JSON output) must NOT be masked.
    assert any(l != -100 for l in labels[-10:])


def test_tokenize_and_mask_unmasked_tokens_decode_to_output_json(real_tokenizer):
    result = tokenize_and_mask(SAMPLE_RECORD, real_tokenizer, max_length=512)
    unmasked_ids = [tok for tok, lab in zip(result["input_ids"], result["labels"]) if lab != -100]
    decoded = real_tokenizer.decode(unmasked_ids)
    assert "two_minute_shutdown" in decoded
    assert "acknowledgement" in decoded


def test_tokenize_and_mask_respects_max_length(real_tokenizer):
    result = tokenize_and_mask(SAMPLE_RECORD, real_tokenizer, max_length=20)
    assert len(result["input_ids"]) <= 20
