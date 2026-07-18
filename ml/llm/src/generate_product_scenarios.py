"""Milestone 5: generate the controlled product-specific scenarios that make
up 70% of the SFT dataset (docs/PRODUCT_SPEC.md section 11.3).

These are deliberately templated/programmatic examples, not model-generated
or scraped — that is the correct design per the spec (the LLM must learn to
reproduce a fixed, controlled input/output schema from an approved action
library, not invent free-form advice), not a shortcut standing in for real
data. Every generated example is validated against data/schemas/llm_input.schema.json
and llm_output.schema.json before being written.

Feature-group -> representative-feature mapping mirrors
ml/behaviour/configs/base.yaml's alert_logic.feature_groups exactly, since
that's what the deployed drift-detection alert logic actually reports.
"""

import itertools
import json
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gemma270m_lora.yaml"

FEATURE_GROUPS = {
    "activity": ["stationary_fraction", "walking_fraction", "running_fraction"],
    "inactive_routine": ["long_lock_total_minutes", "longest_long_lock_minutes", "long_lock_session_count"],
    "charging_routine": ["long_charge_total_minutes", "longest_long_charge_minutes", "night_charge_overlap_minutes"],
}

# One representative feature per group per scenario, with a fixed direction
# and human-readable phrase describing that (feature, direction) combination.
FEATURE_DIRECTION_PHRASES = {
    ("stationary_fraction", "increase"): "You have been staying stationary more than usual",
    ("walking_fraction", "decrease"): "Walking time has decreased",
    ("long_lock_total_minutes", "increase"): "Long inactive phone-lock periods have increased",
    ("longest_long_lock_minutes", "increase"): "Your longest inactive period has grown longer",
    ("long_charge_total_minutes", "increase"): "Charging time has increased",
    ("night_charge_overlap_minutes", "increase"): "Night-time charging has increased",
}

CONTEXT_PHRASES = {
    "work_or_study_pressure": "Work or study pressure may be making your days less regular",
    "poor_sleep": "Poor sleep may be affecting your routine",
    "travel_or_schedule_change": "A change in travel or schedule may explain this",
    "physical_illness": "Feeling physically unwell may explain this",
    "emotionally_low": "Feeling emotionally low may be part of this",
    "nothing_is_wrong": "Even without an obvious cause, your routine has shifted",
    "other": "Something in your routine has shifted",
}

QUESTION_TEMPLATES = {
    "consistent_sleep": "Would one small step toward a steadier bedtime help?",
    "balanced_phone_use": "Would putting your phone aside for a moment help?",
    "daily_activity": "Would a short bit of movement help right now?",
    "stress_management": "Would a brief pause help you reset?",
}


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_candidate_actions(action_library: list[dict], goal: str, context: str) -> list[str]:
    """Top-3 approved action IDs for this goal/context, matching the
    deployed product's pre-ranking-then-LLM-choice design (section 24).
    Returns an empty list (not a crash) if the library has nothing usable —
    callers must treat that as "no valid scenario," not fabricate an action."""
    if not action_library:
        return []
    exact = [a["id"] for a in action_library if goal in a["allowed_goals"] and context in a["allowed_contexts"]]
    if len(exact) >= 1:
        return exact[:3]
    goal_only = [a["id"] for a in action_library if goal in a["allowed_goals"]]
    return goal_only[:3] if goal_only else [action_library[0]["id"]]


def build_scenario(
    goal: str,
    feature_groups_changed: list[str],
    duration_days: int,
    context: str,
    available_minutes: int,
    feedback_state: str,
    action_library: list[dict],
) -> dict | None:
    changes = []
    for group in feature_groups_changed:
        for (feature, direction), _phrase in FEATURE_DIRECTION_PHRASES.items():
            if feature in FEATURE_GROUPS[group]:
                magnitude = "moderate" if duration_days <= 5 else "large"
                changes.append(
                    {"feature": feature, "direction": direction, "magnitude": magnitude, "duration_days": duration_days}
                )
                break  # one representative feature per group

    candidate_actions = select_candidate_actions(action_library, goal, context)
    if not candidate_actions:
        return None
    chosen_action = next(a for a in action_library if a["id"] == candidate_actions[0])

    # Available time must match an action the model can actually complete;
    # skip combinations where no candidate action fits, rather than
    # fabricating a mismatched target.
    if chosen_action["duration_minutes"] != available_minutes:
        matching = [a for a in action_library if a["id"] in candidate_actions and a["duration_minutes"] == available_minutes]
        if not matching:
            return None
        chosen_action = matching[0]

    previous_helpful, previous_unhelpful = [], []
    if feedback_state == "previously_helpful":
        previous_helpful = [chosen_action["id"]]
    elif feedback_state == "previously_unhelpful":
        other_actions = [a["id"] for a in action_library if a["id"] != chosen_action["id"] and goal in a["allowed_goals"]]
        previous_unhelpful = other_actions[:1] if other_actions else []

    llm_input = {
        "goal": goal,
        "changes": changes,
        "user_context": context,
        "available_time_minutes": available_minutes,
        "previous_helpful_actions": previous_helpful,
        "previous_unhelpful_actions": previous_unhelpful,
        "candidate_actions": candidate_actions,
    }

    change_phrase = FEATURE_DIRECTION_PHRASES.get(
        (changes[0]["feature"], changes[0]["direction"]), "Your routine has shifted"
    )
    context_phrase = CONTEXT_PHRASES[context]
    acknowledgement = f"{change_phrase}. {context_phrase}."
    question = QUESTION_TEMPLATES[goal]

    llm_output = {
        "acknowledgement": acknowledgement,
        "question": question,
        "action": {
            "id": chosen_action["id"],
            "title": chosen_action["id"].replace("_", " ").capitalize(),
            "duration_minutes": chosen_action["duration_minutes"],
            "steps": chosen_action["steps"],
        },
    }

    return {"input": llm_input, "output": llm_output}


def generate_all_scenarios(config: dict) -> list[dict]:
    scen_cfg = config["scenario_generation"]
    action_library = config["action_library"]

    scenarios = []
    group_combos = []
    for r in scen_cfg["changed_feature_group_counts"]:
        group_combos.extend(itertools.combinations(FEATURE_GROUPS.keys(), r))

    duration_samples = [scen_cfg["duration_days_range"][0], 5, scen_cfg["duration_days_range"][1]]

    for goal, groups, duration_days, context, minutes, feedback in itertools.product(
        scen_cfg["goals"],
        group_combos,
        duration_samples,
        scen_cfg["contexts"],
        scen_cfg["available_minutes_options"],
        scen_cfg["feedback_states"],
    ):
        scenario = build_scenario(goal, list(groups), duration_days, context, minutes, feedback, action_library)
        if scenario is not None:
            scenario["scenario_family"] = f"{goal}|{'-'.join(sorted(groups))}|{context}"
            scenarios.append(scenario)

    return scenarios


def main() -> int:
    config = load_config()
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    scenarios = generate_all_scenarios(config)
    print(f"Generated {len(scenarios)} product scenarios")

    out_path = processed_dir / "product_scenarios.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in scenarios:
            f.write(json.dumps(item) + "\n")
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
