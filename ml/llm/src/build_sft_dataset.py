"""Milestone 5: combine product scenarios (70%) with EmpatheticDialogues-
derived (15%) and ESConv-derived (15%) examples into the final SFT dataset,
per docs/PRODUCT_SPEC.md section 11.3. Every example is validated against
data/schemas/llm_input.schema.json and llm_output.schema.json before being
written — invalid examples are dropped and counted, never silently kept.

Design decision for the ED/ESConv-derived 30%: those source conversations are
about general life situations, not actual behavioural-drift evidence, so they
cannot supply a real "changes" array — there is no sensor data behind them.
Each is mapped to a plausible, schema-valid MindPulse scenario (context
derived from the conversation's real emotion/problem type, one representative
feature "change" consistent with that context, an action drawn from the real
approved library) so the model learns REALISTIC ACKNOWLEDGEMENT LANGUAGE
STYLE and safe question/action selection from real human dialogue text, while
the structural "changes" evidence is templated rather than claimed to be real
sensor data — the acknowledgement text itself is adapted from the real
dialogue (diversity of phrasing), the question and action are templated/
approved for schema safety. This is standard SFT style-augmentation, not a
claim that these conversations are genuine MindPulse telemetry.
"""

import json
import random
import re
from pathlib import Path

import jsonschema
import yaml

from generate_product_scenarios import CONTEXT_PHRASES, FEATURE_DIRECTION_PHRASES, FEATURE_GROUPS, QUESTION_TEMPLATES, select_candidate_actions

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gemma270m_lora.yaml"
SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "data" / "schemas"

ED_EMOTION_TO_CONTEXT = {
    "annoyed": "work_or_study_pressure",
    "disappointed": "work_or_study_pressure",
    "anxious": "emotionally_low",
    "apprehensive": "emotionally_low",
    "anticipating": "other",
    "content": "nothing_is_wrong",
    "hopeful": "nothing_is_wrong",
    "prepared": "nothing_is_wrong",
    "confident": "nothing_is_wrong",
    "trusting": "nothing_is_wrong",
    "caring": "other",
    "grateful": "nothing_is_wrong",
    "joyful": "nothing_is_wrong",
    "proud": "nothing_is_wrong",
    "faithful": "nothing_is_wrong",
    "impressed": "nothing_is_wrong",
    "excited": "nothing_is_wrong",
    "surprised": "other",
    "sentimental": "other",
    "nostalgic": "other",
}
ESCONV_EMOTION_TO_CONTEXT = {
    "anxiety": "emotionally_low",
    "depression": "emotionally_low",
    "sadness": "emotionally_low",
    "fear": "emotionally_low",
    "anger": "work_or_study_pressure",
    "guilt": "emotionally_low",
    "shame": "emotionally_low",
    "jealousy": "other",
    "nervousness": "work_or_study_pressure",
    "disgust": "other",
    "pain": "physical_illness",
}
DEFAULT_GOAL = "stress_management"
DEFAULT_FEATURE_GROUP = "activity"


def load_schema(name: str) -> dict:
    with open(SCHEMAS_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def smart_truncate(text: str, max_len: int = 200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_period = truncated.rfind(".")
    if last_period > max_len * 0.5:
        return truncated[: last_period + 1]
    last_space = truncated.rfind(" ")
    return (truncated[:last_space] if last_space > 0 else truncated).rstrip() + "."


def dialogue_derived_example(
    source: str, acknowledgement_text: str, context: str, action_library: list[dict], conv_key: str | int = ""
) -> dict | None:
    goal = DEFAULT_GOAL
    candidate_actions = select_candidate_actions(action_library, goal, context)
    if not candidate_actions:
        return None
    action = next((a for a in action_library if a["id"] == candidate_actions[0]), None)
    if action is None:
        return None

    feature = FEATURE_GROUPS[DEFAULT_FEATURE_GROUP][0]
    direction = "increase"
    changes = [{"feature": feature, "direction": direction, "magnitude": "moderate", "duration_days": 4}]

    llm_input = {
        "goal": goal,
        "changes": changes,
        "user_context": context,
        "available_time_minutes": action["duration_minutes"],
        "previous_helpful_actions": [],
        "previous_unhelpful_actions": [],
        "candidate_actions": candidate_actions,
    }
    llm_output = {
        "acknowledgement": smart_truncate(acknowledgement_text),
        "question": QUESTION_TEMPLATES[goal],
        "action": {
            "id": action["id"],
            "title": action["id"].replace("_", " ").capitalize(),
            "duration_minutes": action["duration_minutes"],
            "steps": action["steps"],
        },
    }
    # Each dialogue-derived example comes from a DISTINCT real conversation —
    # unlike product scenarios (deliberate variants of one underlying
    # scenario), two ED/ESConv conversations sharing a mapped context are not
    # paraphrases of each other. Grouping all of them into one family by
    # context alone (an earlier version of this function) produced a handful
    # of huge families (~98 examples each) that skewed the train/val/test
    # split badly (72/8.7/19% instead of the intended 80/10/10, confirmed by
    # running the real pipeline) — each conversation gets its own family here.
    family = f"{source}|{context}|{conv_key}" if conv_key != "" else f"{source}|{context}"
    return {"input": llm_input, "output": llm_output, "scenario_family": family}


def build_ed_examples(filtered_path: Path, action_library: list[dict], target_count: int, seed: int) -> list[dict]:
    conversations = [json.loads(line) for line in open(filtered_path, "r", encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(conversations)

    examples = []
    for conv in conversations:
        if len(examples) >= target_count:
            break
        context = ED_EMOTION_TO_CONTEXT.get(conv["context"], "other")
        responder_turns = [t["utterance"] for t in conv["turns"][1:]]
        if not responder_turns:
            continue
        example = dialogue_derived_example(
            "empathetic_dialogues", responder_turns[0], context, action_library, conv_key=conv["conv_id"]
        )
        if example is not None:
            examples.append(example)
    return examples


def build_esconv_examples(filtered_path: Path, action_library: list[dict], target_count: int, seed: int) -> list[dict]:
    conversations = [json.loads(line) for line in open(filtered_path, "r", encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(conversations)

    examples = []
    for conv in conversations:
        if len(examples) >= target_count:
            break
        context = ESCONV_EMOTION_TO_CONTEXT.get(conv.get("emotion_type"), "other")
        supporter_turns = [t["content"] for t in conv["turns"] if t["speaker"] == "supporter"]
        if not supporter_turns:
            continue
        example = dialogue_derived_example(
            "esconv", supporter_turns[0], context, action_library, conv_key=conv["conv_id"]
        )
        if example is not None:
            examples.append(example)
    return examples


def validate_example(example: dict, input_schema: dict, output_schema: dict) -> bool:
    try:
        jsonschema.validate(example["input"], input_schema)
        jsonschema.validate(example["output"], output_schema)
        return True
    except jsonschema.ValidationError:
        return False


def split_by_family(examples: list[dict], seed: int, train_frac: float, val_frac: float) -> dict:
    families = sorted({e["scenario_family"] for e in examples})
    rng = random.Random(seed)
    rng.shuffle(families)

    n = len(families)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_families = set(families[:n_train])
    val_families = set(families[n_train : n_train + n_val])

    splits = {"train": [], "validation": [], "test": []}
    for e in examples:
        if e["scenario_family"] in train_families:
            splits["train"].append(e)
        elif e["scenario_family"] in val_families:
            splits["validation"].append(e)
        else:
            splits["test"].append(e)
    return splits


def main() -> int:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    action_library = config["action_library"]
    seed = config["seed"]

    product_path = processed_dir / "product_scenarios.jsonl"
    ed_path = processed_dir / "empathetic_dialogues_filtered.jsonl"
    esconv_path = processed_dir / "esconv_filtered.jsonl"
    for p in [product_path, ed_path, esconv_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run the upstream generator/filter scripts first.")
            return 1

    product_scenarios = [json.loads(line) for line in open(product_path, "r", encoding="utf-8")]
    n_product = len(product_scenarios)
    composition = config["sft_dataset"]["composition"]
    n_ed = int(n_product / composition["product_scenarios"] * composition["empathetic_dialogues"])
    n_esconv = int(n_product / composition["product_scenarios"] * composition["esconv"])

    ed_examples = build_ed_examples(ed_path, action_library, n_ed, seed)
    esconv_examples = build_esconv_examples(esconv_path, action_library, n_esconv, seed)
    print(f"product_scenarios={n_product}, ed_derived={len(ed_examples)} (target {n_ed}), esconv_derived={len(esconv_examples)} (target {n_esconv})")

    all_examples = product_scenarios + ed_examples + esconv_examples

    input_schema = load_schema("llm_input.schema.json")
    output_schema = load_schema("llm_output.schema.json")
    valid_examples = [e for e in all_examples if validate_example(e, input_schema, output_schema)]
    n_invalid = len(all_examples) - len(valid_examples)
    print(f"Schema validation: {len(valid_examples)} valid, {n_invalid} invalid (dropped)")

    split_cfg = config["sft_dataset"]["split"]
    splits = split_by_family(valid_examples, seed, split_cfg["train"], split_cfg["validation"])
    print(f"Split: train={len(splits['train'])}, validation={len(splits['validation'])}, test={len(splits['test'])}")

    out_path = processed_dir / "mindpulse_sft.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for split_name, items in splits.items():
            for item in items:
                record = {"split": split_name, "input": item["input"], "output": item["output"]}
                f.write(json.dumps(record) + "\n")

    print(f"Saved {len(valid_examples)} examples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
