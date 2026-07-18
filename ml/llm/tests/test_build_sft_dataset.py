"""Unit tests for SFT dataset assembly: truncation, dialogue-to-example
conversion, schema validation, and family-aware splitting."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_sft_dataset import dialogue_derived_example, load_schema, smart_truncate, split_by_family, validate_example  # noqa: E402

ACTION_LIBRARY = [
    {
        "id": "one_minute_breathing",
        "allowed_goals": ["stress_management"],
        "allowed_contexts": ["work_or_study_pressure", "other", "emotionally_low", "nothing_is_wrong", "physical_illness"],
        "duration_minutes": 1,
        "steps": ["Breathe in.", "Breathe out."],
    }
]


def test_smart_truncate_short_text_unchanged():
    assert smart_truncate("Short text.") == "Short text."


def test_smart_truncate_cuts_at_sentence_boundary():
    text = "This is sentence one. " + "x" * 190 + ". More text after."
    result = smart_truncate(text, max_len=50)
    assert len(result) <= 51
    assert result.endswith(".")


def test_smart_truncate_collapses_whitespace():
    assert smart_truncate("a   b\n\nc") == "a b c"


def test_dialogue_derived_example_is_schema_shaped():
    example = dialogue_derived_example("empathetic_dialogues", "That sounds really tiring.", "work_or_study_pressure", ACTION_LIBRARY)
    assert example is not None
    assert example["input"]["goal"] == "stress_management"
    assert example["output"]["action"]["id"] == "one_minute_breathing"
    assert example["output"]["action"]["duration_minutes"] == 1
    assert example["input"]["available_time_minutes"] == 1


def test_dialogue_derived_example_conv_key_produces_distinct_families():
    ex1 = dialogue_derived_example("empathetic_dialogues", "text one", "work_or_study_pressure", ACTION_LIBRARY, conv_key="conv_A")
    ex2 = dialogue_derived_example("empathetic_dialogues", "text two", "work_or_study_pressure", ACTION_LIBRARY, conv_key="conv_B")
    assert ex1["scenario_family"] != ex2["scenario_family"]


def test_dialogue_derived_example_returns_none_for_unmatched_action():
    assert dialogue_derived_example("empathetic_dialogues", "text", "work_or_study_pressure", []) is None


def test_validate_example_real_schemas_accept_valid_example():
    input_schema = load_schema("llm_input.schema.json")
    output_schema = load_schema("llm_output.schema.json")
    example = dialogue_derived_example("empathetic_dialogues", "That sounds tiring.", "work_or_study_pressure", ACTION_LIBRARY)
    assert validate_example(example, input_schema, output_schema) is True


def test_validate_example_rejects_invalid_action_id():
    input_schema = load_schema("llm_input.schema.json")
    output_schema = load_schema("llm_output.schema.json")
    example = dialogue_derived_example("empathetic_dialogues", "text", "work_or_study_pressure", ACTION_LIBRARY)
    example["output"]["action"]["id"] = "not_an_approved_action"
    assert validate_example(example, input_schema, output_schema) is False


def test_split_by_family_keeps_same_family_together():
    examples = [
        {"scenario_family": "fam1", "id": 1}, {"scenario_family": "fam1", "id": 2},
        {"scenario_family": "fam2", "id": 3}, {"scenario_family": "fam3", "id": 4},
        {"scenario_family": "fam4", "id": 5}, {"scenario_family": "fam5", "id": 6},
        {"scenario_family": "fam6", "id": 7}, {"scenario_family": "fam7", "id": 8},
        {"scenario_family": "fam8", "id": 9}, {"scenario_family": "fam9", "id": 10},
    ]
    splits = split_by_family(examples, seed=42, train_frac=0.8, val_frac=0.1)
    all_ids = set()
    for split_items in splits.values():
        all_ids |= {e["id"] for e in split_items}
    assert all_ids == {e["id"] for e in examples}
    # fam1's two examples must land in the same split.
    fam1_splits = [name for name, items in splits.items() if any(e["scenario_family"] == "fam1" for e in items)]
    assert len(fam1_splits) == 1
