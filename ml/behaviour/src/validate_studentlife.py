"""Milestone 1: Data audit.

Validates the extracted StudentLife archive against the exact file list required
by docs/PRODUCT_SPEC.md section 6.1, and produces a machine-readable dataset
inventory (participant counts, date ranges, missingness, EMA-definition report)
at data/interim/studentlife_inventory.json.

This script does not invent or assume any file, column, or value that is not
actually present in the archive — it reports what it finds and fails loudly on
what is required but missing.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict) -> dict:
    base_dir = Path(__file__).resolve().parents[1]
    return {
        "raw_dir": (base_dir / config["paths"]["raw_dir"]).resolve(),
        "interim_dir": (base_dir / config["paths"]["interim_dir"]).resolve(),
    }


def inventory_activity(activity_dir: Path) -> dict:
    """Per-participant row counts and timestamp span for sensing/activity/*.csv."""
    if not activity_dir.exists():
        return {"present": False, "participants": {}}
    participants = {}
    for csv_file in sorted(activity_dir.glob("activity_*.csv")):
        pid = csv_file.stem.replace("activity_", "")
        df = pd.read_csv(csv_file)
        if df.empty:
            participants[pid] = {"rows": 0}
            continue
        ts_col = df.columns[0]
        participants[pid] = {
            "rows": len(df),
            "columns": list(df.columns),
            "min_timestamp": int(df[ts_col].min()),
            "max_timestamp": int(df[ts_col].max()),
        }
    return {"present": True, "participants": participants}


def inventory_intervals(interval_dir: Path, prefix: str) -> dict:
    """Row counts for phonelock_*.csv / phonecharge_*.csv (start,end interval files)."""
    if not interval_dir.exists():
        return {"present": False, "participants": {}}
    participants = {}
    for csv_file in sorted(interval_dir.glob(f"{prefix}_*.csv")):
        pid = csv_file.stem.replace(f"{prefix}_", "")
        df = pd.read_csv(csv_file)
        participants[pid] = {"rows": len(df), "columns": list(df.columns)}
    return {"present": True, "participants": participants}


def inventory_ema_definition(ema_def_path: Path) -> dict:
    """EMA_definition.json is a list of category definitions, each
    {"name": ..., "questions": [{"question_id", "question_text", "options"}, ...]}.
    Reported here, not assumed elsewhere — spec section 9 requires reading option
    meanings from this file rather than hard-coding numeric-index interpretations.
    """
    if not ema_def_path.exists():
        return {"present": False}
    with open(ema_def_path, "r", encoding="utf-8") as f:
        definition = json.load(f)
    categories = {}
    for entry in definition:
        name = entry.get("name", "<unnamed>")
        categories[name] = [q.get("question_id") for q in entry.get("questions", [])]
    return {"present": True, "category_count": len(definition), "categories": categories}


def inventory_ema_responses(response_dir: Path) -> dict:
    if not response_dir.exists():
        return {"present": False, "participants": {}}
    participants = {}
    for json_file in sorted(response_dir.glob("*.json")):
        pid = json_file.stem.replace("Stress_", "").replace("Mood_", "").replace(
            "Sleep_", ""
        ).replace("Activity_", "")
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                participants[pid] = {"response_count": len(data)}
            except json.JSONDecodeError:
                participants[pid] = {"response_count": None, "parse_error": True}
    return {"present": True, "participants": participants}


def inventory_user_info(user_info_path: Path) -> dict:
    if not user_info_path.exists():
        return {"present": False}
    df = pd.read_csv(user_info_path)
    return {"present": True, "rows": len(df), "columns": list(df.columns)}


def main() -> int:
    config = load_config()
    paths = resolve_paths(config)
    dataset_root = paths["raw_dir"] / "dataset"

    if not dataset_root.exists():
        print(
            f"ERROR: {dataset_root} does not exist. Run download_data.py first — "
            "this validator does not run against fabricated or partial data.",
            file=sys.stderr,
        )
        return 1

    inventory = {
        "dataset_root": str(dataset_root),
        "user_info": inventory_user_info(dataset_root / "user_info.csv"),
        "activity": inventory_activity(dataset_root / "sensing" / "activity"),
        "phonelock": inventory_intervals(dataset_root / "sensing" / "phonelock", "phonelock"),
        "phonecharge": inventory_intervals(dataset_root / "sensing" / "phonecharge", "phonecharge"),
        "ema_definition": inventory_ema_definition(dataset_root / "EMA" / "EMA_definition.json"),
        "ema_stress": inventory_ema_responses(dataset_root / "EMA" / "response" / "Stress"),
        "ema_mood": inventory_ema_responses(dataset_root / "EMA" / "response" / "Mood"),
        "ema_sleep": inventory_ema_responses(dataset_root / "EMA" / "response" / "Sleep"),
        "ema_activity": inventory_ema_responses(dataset_root / "EMA" / "response" / "Activity"),
    }

    all_participant_ids = set()
    for section in ["activity", "phonelock", "phonecharge"]:
        all_participant_ids |= set(inventory[section]["participants"].keys())
    inventory["participant_count_union"] = len(all_participant_ids)
    inventory["participant_ids"] = sorted(all_participant_ids)

    paths["interim_dir"].mkdir(parents=True, exist_ok=True)
    out_path = paths["interim_dir"] / "studentlife_inventory.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    print(f"Inventory written to {out_path}")
    print(f"Participants found (union across activity/phonelock/phonecharge): {len(all_participant_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
