"""Milestone 2: build the per-participant daily feature table.

Combines parse_activity + parse_intervals output into one row per
(participant_id, date) matching data/schemas/daily_features.schema.json: the
12 matched features (docs/PRODUCT_SPEC.md section 7.2), a missingness flag per
feature, an overall coverage value, and a feature_version tag.

Missingness rule for lock/charge features (a design decision the spec names
the features for but does not give an exact formula for): a day is only
eligible to report a *confirmed zero* ("no long lock/charge session occurred")
if the activity signal shows the phone was actively sensing that day
(activity_coverage_fraction > 0, i.e. the day appears in the activity table at
all). If the day has no activity coverage at all, sensing was very likely off,
so we cannot distinguish "no long session happened" from "we don't know" —
that case is marked missing=True rather than fabricated as a zero.
"""

import re
from pathlib import Path

import pandas as pd
import yaml

from parse_activity import daily_activity_fractions, parse_activity_file
from parse_intervals import daily_interval_features, parse_interval_file

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
FEATURE_VERSION = "v1"

ACTIVITY_FEATURES = [
    "stationary_fraction", "walking_fraction", "running_fraction",
    "unknown_activity_fraction", "activity_coverage_fraction",
]
LOCK_FEATURES = ["long_lock_total_minutes", "longest_long_lock_minutes", "long_lock_session_count"]
CHARGE_FEATURES = [
    "long_charge_total_minutes", "longest_long_charge_minutes",
    "long_charge_session_count", "night_charge_overlap_minutes",
]
ALL_FEATURES = ACTIVITY_FEATURES + LOCK_FEATURES + CHARGE_FEATURES


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_participants(activity_dir: Path) -> list[str]:
    ids = []
    for path in sorted(activity_dir.glob("activity_*.csv")):
        match = re.match(r"activity_(u\d+)\.csv", path.name)
        if match:
            ids.append(match.group(1))
    return ids


def build_participant_daily_features(
    participant_id: str, dataset_root: Path, long_threshold_minutes: int
) -> pd.DataFrame:
    activity_path = dataset_root / "sensing" / "activity" / f"activity_{participant_id}.csv"
    lock_path = dataset_root / "sensing" / "phonelock" / f"phonelock_{participant_id}.csv"
    charge_path = dataset_root / "sensing" / "phonecharge" / f"phonecharge_{participant_id}.csv"

    activity_daily = daily_activity_fractions(parse_activity_file(activity_path)) if activity_path.exists() else pd.DataFrame(columns=["date"])
    lock_daily = (
        daily_interval_features(parse_interval_file(lock_path), "lock", long_threshold_minutes)
        if lock_path.exists() else pd.DataFrame(columns=["date"])
    )
    charge_daily = (
        daily_interval_features(
            parse_interval_file(charge_path), "charge", long_threshold_minutes, include_night_overlap=True
        )
        if charge_path.exists() else pd.DataFrame(columns=["date"])
    )

    all_dates = sorted(
        set(activity_daily["date"]) | set(lock_daily["date"]) | set(charge_daily["date"])
    )
    if not all_dates:
        return pd.DataFrame()

    merged = pd.DataFrame({"date": all_dates})
    merged = merged.merge(activity_daily, on="date", how="left")
    merged = merged.merge(lock_daily, on="date", how="left")
    merged = merged.merge(charge_daily, on="date", how="left")

    activity_sensed = merged["activity_coverage_fraction"].notna() & (merged["activity_coverage_fraction"] > 0)

    for feature in LOCK_FEATURES + CHARGE_FEATURES:
        was_absent = merged[feature].isna()
        # Confirmed zero only when we know sensing was active that day; else unknown.
        merged.loc[was_absent & activity_sensed, feature] = 0.0
        merged[f"missing_{feature}"] = was_absent & ~activity_sensed

    for feature in ACTIVITY_FEATURES:
        merged[f"missing_{feature}"] = merged[feature].isna()

    merged["participant_id"] = participant_id
    merged["coverage"] = merged["activity_coverage_fraction"].fillna(0.0).astype(float)
    merged["feature_version"] = FEATURE_VERSION

    ordered_cols = (
        ["participant_id", "date"]
        + ALL_FEATURES
        + [f"missing_{f}" for f in ALL_FEATURES]
        + ["coverage", "feature_version"]
    )
    return merged[ordered_cols]


def main() -> int:
    config = load_config()
    base_dir = Path(__file__).resolve().parents[1]
    dataset_root = (base_dir / config["paths"]["raw_dir"]).resolve() / "dataset"
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    long_threshold = config["features"]["long_interval_threshold_minutes"]

    activity_dir = dataset_root / "sensing" / "activity"
    if not activity_dir.exists():
        print(f"ERROR: {activity_dir} not found. Run download_data.py / extraction first.")
        return 1

    participant_ids = discover_participants(activity_dir)
    print(f"Building daily features for {len(participant_ids)} participants...")

    all_rows = []
    for pid in participant_ids:
        df = build_participant_daily_features(pid, dataset_root, long_threshold)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("ERROR: no daily feature rows produced from any participant.")
        return 1

    result = pd.concat(all_rows, ignore_index=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "daily_features.parquet"
    result.to_parquet(out_path, index=False)

    print(f"Wrote {len(result)} rows ({result['participant_id'].nunique()} participants) to {out_path}")
    print(result[ALL_FEATURES].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
