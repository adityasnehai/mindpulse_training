"""Unit tests for the independent SFT dataset validation pass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from validate_sft_dataset import load_schema, validate_records, write_manual_review_sample  # noqa: E402

VALID_RECORD = {
    "split": "train",
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
        "acknowledgement": "Your routine has shifted a little.",
        "question": "Would one small step help?",
        "action": {
            "id": "two_minute_shutdown",
            "title": "Two minute shutdown",
            "duration_minutes": 2,
            "steps": ["Step one.", "Step two."],
        },
    },
}


def test_valid_record_passes_all_checks():
    input_schema = load_schema("llm_input.schema.json")
    output_schema = load_schema("llm_output.schema.json")
    result = validate_records([VALID_RECORD], input_schema, output_schema, {"two_minute_shutdown"})
    assert result["n_schema_failures"] == 0
    assert result["n_action_id_failures"] == 0
    assert result["n_keyword_failures"] == 0


def test_unapproved_action_id_is_flagged():
    input_schema = load_schema("llm_input.schema.json")
    output_schema = load_schema("llm_output.schema.json")
    result = validate_records([VALID_RECORD], input_schema, output_schema, {"some_other_action"})
    assert result["n_action_id_failures"] == 1


def test_keyword_violation_in_output_text_is_flagged():
    import copy

    bad_record = copy.deepcopy(VALID_RECORD)
    bad_record["output"]["acknowledgement"] = "You should see a psychiatrist about this."
    input_schema = load_schema("llm_input.schema.json")
    output_schema = load_schema("llm_output.schema.json")
    result = validate_records([bad_record], input_schema, output_schema, {"two_minute_shutdown"})
    assert result["n_keyword_failures"] == 1


def test_malformed_input_is_flagged_as_schema_failure():
    import copy

    bad_record = copy.deepcopy(VALID_RECORD)
    bad_record["input"]["goal"] = "not_a_real_goal"
    input_schema = load_schema("llm_input.schema.json")
    output_schema = load_schema("llm_output.schema.json")
    result = validate_records([bad_record], input_schema, output_schema, {"two_minute_shutdown"})
    assert result["n_schema_failures"] == 1
    assert result["schema_valid"] == 0


def test_manual_review_sample_respects_caps(tmp_path):
    records = (
        [{**VALID_RECORD, "split": "train"} for _ in range(400)]
        + [{**VALID_RECORD, "split": "validation"} for _ in range(50)]
    )
    out_path = tmp_path / "sample.jsonl"
    write_manual_review_sample(records, out_path, seed=42)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 300 + 50  # capped at 300 train, but validation has only 50 available
