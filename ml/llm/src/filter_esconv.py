"""Milestone 5: filter the real ESConv dataset per docs/PRODUCT_SPEC.md
section 6.3.

Verified against the real ESConv.json: supporter turns are annotated with one
of 8 real strategies ('Question', 'Reflection of feelings', 'Restatement or
Paraphrasing', 'Providing Suggestions', 'Information', 'Affirmation and
Reassurance', 'Self-disclosure', 'Others'). The spec names exactly five to
keep (Questions, Reflection of feelings, Restatement or paraphrasing,
Providing suggestions, Information) and explicitly excludes Self-disclosure —
'Affirmation and Reassurance' and 'Others' are not in the spec's keep list,
so they are excluded too, conservatively (not explicitly named as safe).

Conversation-level exclusion: `problem_type == "Alcohol Abuse"` (explicitly
named in section 6.3) is dropped entirely, plus the shared keyword safety
filter is applied to every kept turn.
"""

import json
from pathlib import Path

import yaml

from safety_filters import contains_excluded_content

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gemma270m_lora.yaml"

ALLOWED_STRATEGIES = {
    "Question",
    "Reflection of feelings",
    "Restatement or Paraphrasing",
    "Providing Suggestions",
    "Information",
}
EXCLUDED_PROBLEM_TYPES = {"Alcohol Abuse"}


def filter_conversations(conversations: list[dict]) -> list[dict]:
    """Returns a list of {conv_id, emotion_type, problem_type, situation,
    turns: [{speaker, strategy, content}]} keeping only supporter turns whose
    strategy is allowed (seeker turns are kept as context) and only
    conversations whose problem_type isn't excluded and whose content passes
    the keyword safety filter."""
    kept = []
    for idx, conv in enumerate(conversations):
        if conv.get("problem_type") in EXCLUDED_PROBLEM_TYPES:
            continue
        situation = conv.get("situation", "")
        if contains_excluded_content(situation):
            continue

        turns = []
        conversation_safe = True
        for turn in conv.get("dialog", []):
            content = turn.get("content", "")
            if contains_excluded_content(content):
                conversation_safe = False
                break
            speaker = turn.get("speaker")
            strategy = turn.get("annotation", {}).get("strategy")
            if speaker == "supporter" and strategy not in ALLOWED_STRATEGIES:
                continue  # drop this turn, keep the rest of the conversation
            turns.append({"speaker": speaker, "strategy": strategy, "content": content})

        if not conversation_safe or not turns:
            continue

        kept.append(
            {
                "conv_id": idx,
                "emotion_type": conv.get("emotion_type"),
                "problem_type": conv.get("problem_type"),
                "situation": situation,
                "turns": turns,
            }
        )
    return kept


def main() -> int:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = (base_dir / config["paths"]["raw_dir"]).resolve()
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    esconv_path = raw_dir / "ESConv.json"
    if not esconv_path.exists():
        print(f"ERROR: {esconv_path} not found. Run download_dialogue_data.py first.")
        return 1

    with open(esconv_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    kept = filter_conversations(conversations)
    print(f"{len(conversations)} conversations -> {len(kept)} kept")

    out_path = processed_dir / "esconv_filtered.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item) + "\n")

    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
