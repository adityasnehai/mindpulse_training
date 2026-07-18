"""Milestone 5: filter the real EmpatheticDialogues dataset per
docs/PRODUCT_SPEC.md section 6.2.

The spec names five categories to keep by DESCRIPTION ("acknowledging
frustration," "tiredness," "uncertainty," "neutral non-clinical validation,"
"gentle follow-up questions") but EmpatheticDialogues' real `context` column
uses its own fixed set of 32 emotion labels (verified against the actual
downloaded train.csv), which don't literally contain those words. ALLOWED_
CONTEXTS below is a considered, documented mapping from the spec's intent to
the real label set — not fabricated, but an interpretive judgment call, exactly
like the Stress-EMA severity mapping in ml/behaviour/src/ema_labels.py.
Excluded emotions (devastated, terrified, furious, guilty, ashamed, lonely,
sad, jealous, angry, afraid, disgusted, embarrassed) are higher-intensity or
edge toward clinical territory — conservatively dropped even though a few
individual conversations within them might have been fine, because
docs/PRODUCT_SPEC.md's explicit priority is "do not train the model to
imitate an ongoing therapist-client conversation" and MindPulse's actual
outputs are meant to be mild and routine, not deep emotional support.

A keyword-level safety filter (safety_filters.py) is applied on top as
defense-in-depth, since even a "content"-tagged conversation could
occasionally mention something that should be excluded.
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from safety_filters import contains_excluded_content

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gemma270m_lora.yaml"

# Kept: mild frustration/tiredness-adjacent, uncertainty-adjacent, and neutral/
# positive validation-adjacent emotions. See module docstring for reasoning.
ALLOWED_CONTEXTS = {
    "annoyed",  # frustration
    "disappointed",  # mild frustration
    "anxious", "apprehensive", "anticipating",  # uncertainty
    "content", "hopeful", "prepared", "confident", "trusting", "caring",
    "grateful", "joyful", "proud", "faithful", "impressed", "excited",
    "surprised", "sentimental", "nostalgic",  # neutral / positive validation
}


def unescape(text: str) -> str:
    """EmpatheticDialogues' CSV export escapes literal commas as "_comma_"."""
    if not isinstance(text, str):
        return text
    return text.replace("_comma_", ",")


def load_conversations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
    df["prompt"] = df["prompt"].apply(unescape)
    df["utterance"] = df["utterance"].apply(unescape)
    return df


def filter_conversations(df: pd.DataFrame) -> list[dict]:
    """Returns a list of {conv_id, context, situation, turns: [{speaker_idx, utterance}]}
    for conversations that pass both the emotion-category allowlist and the
    keyword safety filter on every turn."""
    kept = []
    for conv_id, group in df.groupby("conv_id"):
        group = group.sort_values("utterance_idx")
        context = group.iloc[0]["context"]
        if context not in ALLOWED_CONTEXTS:
            continue

        situation = group.iloc[0]["prompt"]
        turns = group[["speaker_idx", "utterance"]].to_dict("records")

        if contains_excluded_content(situation) or any(contains_excluded_content(t["utterance"]) for t in turns):
            continue

        kept.append({"conv_id": conv_id, "context": context, "situation": situation, "turns": turns})
    return kept


def main() -> int:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = (base_dir / config["paths"]["raw_dir"]).resolve()
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_kept = []
    for split in ["train", "valid", "test"]:
        csv_path = raw_dir / "empatheticdialogues" / f"{split}.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found. Run download_dialogue_data.py first.")
            return 1
        df = load_conversations(csv_path)
        kept = filter_conversations(df)
        print(f"{split}: {df['conv_id'].nunique()} conversations -> {len(kept)} kept")
        for item in kept:
            item["split"] = split
        all_kept.extend(kept)

    out_path = processed_dir / "empathetic_dialogues_filtered.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_kept:
            f.write(json.dumps(item) + "\n")

    print(f"Total kept: {len(all_kept)} conversations -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
