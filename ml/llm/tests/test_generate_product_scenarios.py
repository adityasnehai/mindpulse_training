"""Unit tests for the controlled product-scenario generator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_product_scenarios import build_scenario, select_candidate_actions  # noqa: E402

ACTION_LIBRARY = [
    {
        "id": "two_minute_shutdown",
        "allowed_goals": ["consistent_sleep", "stress_management"],
        "allowed_contexts": ["work_or_study_pressure", "poor_sleep"],
        "duration_minutes": 2,
        "steps": ["step1", "step2"],
    },
    {
        "id": "one_minute_breathing",
        "allowed_goals": ["stress_management"],
        "allowed_contexts": ["work_or_study_pressure"],
        "duration_minutes": 1,
        "steps": ["breathe"],
    },
    {
        "id": "five_minute_walk",
        "allowed_goals": ["daily_activity"],
        "allowed_contexts": ["nothing_is_wrong"],
        "duration_minutes": 5,
        "steps": ["walk"],
    },
]


def test_select_candidate_actions_exact_match():
    result = select_candidate_actions(ACTION_LIBRARY, "stress_management", "work_or_study_pressure")
    assert set(result) <= {"two_minute_shutdown", "one_minute_breathing"}
    assert len(result) >= 1


def test_select_candidate_actions_falls_back_to_goal_only():
    result = select_candidate_actions(ACTION_LIBRARY, "daily_activity", "poor_sleep")
    assert "five_minute_walk" in result


def test_build_scenario_matches_available_time_to_action_duration():
    scenario = build_scenario(
        goal="stress_management",
        feature_groups_changed=["activity"],
        duration_days=4,
        context="work_or_study_pressure",
        available_minutes=1,
        feedback_state="no_prior_feedback",
        action_library=ACTION_LIBRARY,
    )
    assert scenario is not None
    assert scenario["output"]["action"]["duration_minutes"] == 1
    assert scenario["input"]["available_time_minutes"] == 1


def test_build_scenario_returns_none_when_no_action_fits_time():
    scenario = build_scenario(
        goal="daily_activity",
        feature_groups_changed=["activity"],
        duration_days=4,
        context="nothing_is_wrong",
        available_minutes=2,  # only five_minute_walk (5 min) fits this goal/context
        feedback_state="no_prior_feedback",
        action_library=ACTION_LIBRARY,
    )
    assert scenario is None


def test_build_scenario_previously_helpful_recorded():
    scenario = build_scenario(
        goal="stress_management",
        feature_groups_changed=["activity"],
        duration_days=4,
        context="work_or_study_pressure",
        available_minutes=2,
        feedback_state="previously_helpful",
        action_library=ACTION_LIBRARY,
    )
    assert scenario is not None
    assert scenario["output"]["action"]["id"] in scenario["input"]["previous_helpful_actions"]


def test_build_scenario_changes_reflect_requested_groups():
    scenario = build_scenario(
        goal="consistent_sleep",
        feature_groups_changed=["inactive_routine", "charging_routine"],
        duration_days=4,
        context="poor_sleep",
        available_minutes=2,
        feedback_state="no_prior_feedback",
        action_library=ACTION_LIBRARY,
    )
    assert scenario is not None
    changed_features = {c["feature"] for c in scenario["input"]["changes"]}
    assert changed_features & {"long_lock_total_minutes", "longest_long_lock_minutes"}
    assert changed_features & {"long_charge_total_minutes", "night_charge_overlap_minutes"}


def test_build_scenario_output_action_id_is_from_approved_library():
    approved_ids = {a["id"] for a in ACTION_LIBRARY}
    scenario = build_scenario(
        goal="stress_management",
        feature_groups_changed=["activity"],
        duration_days=3,
        context="work_or_study_pressure",
        available_minutes=1,
        feedback_state="no_prior_feedback",
        action_library=ACTION_LIBRARY,
    )
    assert scenario is not None
    assert scenario["output"]["action"]["id"] in approved_ids
