"""Milestone 7: desktop inference test on the real converted .litertlm file
(docs/PRODUCT_SPEC.md section 15's "Desktop inference test" deliverable).
Loads the actual exported model via litert_lm.Engine and runs a real
MindPulse-formatted prompt through it, checking the output still parses as
valid structured JSON with an approved action — the same bar Milestone 6's
evaluate_llm.py already cleared for the pre-conversion model, now checked
post-conversion since quantization/export could in principle change behavior.
"""

import json
import re
import sys
from pathlib import Path

import litert_lm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_format import build_user_turn  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gemma270m_lora.yaml"


def extract_json_object(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def main() -> int:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = Path(__file__).resolve().parents[1]
    artifacts_dir = (base_dir / config["paths"]["artifacts_dir"]).resolve()
    model_path = artifacts_dir / "litertlm_output" / "model.litertlm"

    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run convert_litertlm.py first.")
        return 1

    approved_action_ids = {a["id"] for a in config["action_library"]}

    test_inputs = [
        {
            "goal": "stress_management",
            "changes": [{"feature": "stationary_fraction", "direction": "increase", "magnitude": "moderate", "duration_days": 4}],
            "user_context": "work_or_study_pressure",
            "available_time_minutes": 1,
            "previous_helpful_actions": [],
            "previous_unhelpful_actions": [],
            "candidate_actions": ["one_minute_breathing", "short_grounding"],
        },
        {
            "goal": "consistent_sleep",
            "changes": [{"feature": "night_charge_overlap_minutes", "direction": "increase", "magnitude": "large", "duration_days": 5}],
            "user_context": "poor_sleep",
            "available_time_minutes": 5,
            "previous_helpful_actions": [],
            "previous_unhelpful_actions": [],
            "candidate_actions": ["prepare_sleep_environment"],
        },
    ]

    print(f"Loading real .litertlm model from {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)...")
    engine = litert_lm.Engine(str(model_path))

    all_passed = True
    for i, llm_input in enumerate(test_inputs):
        conversation = engine.create_conversation()
        prompt = build_user_turn(llm_input)
        response = conversation.send_message(prompt)

        # The real response is a Message-shaped object/dict:
        # {'role': 'assistant', 'content': [{'type': 'text', 'text': '...'}]}
        # — confirmed by inspecting the actual printed object, not assumed.
        # str(response) on the whole wrapper gives Python repr (single-quoted
        # dict syntax), which is not valid JSON even though the nested
        # "text" field is; extract that field directly instead.
        content = response["content"] if isinstance(response, dict) else response.content
        first_block = content[0]
        raw_text = first_block["text"] if isinstance(first_block, dict) else first_block.text

        print(f"\n--- Test {i + 1} ---")
        print(f"raw output: {raw_text}")

        parsed = extract_json_object(raw_text)
        if parsed is None:
            print("FAIL: could not parse JSON from output")
            all_passed = False
            continue

        action_id = parsed.get("action", {}).get("id")
        if action_id not in approved_action_ids:
            print(f"FAIL: action id {action_id!r} not in approved library")
            all_passed = False
            continue

        print(f"PASS: valid JSON, approved action '{action_id}'")

    engine.close()

    if not all_passed:
        print("\nSome real inference checks FAILED.")
        return 1

    print("\nAll real .litertlm inference checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
