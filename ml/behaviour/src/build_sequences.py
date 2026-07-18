"""Milestone 3 prerequisite: construct 7-day sequence windows from the daily
feature table (docs/PRODUCT_SPEC.md section 7.2/8.2 — model input shape [7, 24]:
12 feature values then 12 missingness indicators, in the same feature order).

Design decision: a window is only built from 7 REAL, CONSECUTIVE calendar-date
rows already present in daily_features.parquet — no calendar day is
synthesized or fabricated to fill a gap. A participant with a gap simply
cannot start a window at a date whose 7-day span crosses that gap. This is
more conservative than filling gaps with a placeholder "fully missing" row,
but it means every value fed to the model traces back to an actual observed
day, never an invented one. Individual *feature* missingness within an
observed day (e.g. no long-lock session that day) is still tracked via the
missing_* columns and is not the same as a missing day.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from build_daily_features import ALL_FEATURES

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
WINDOW_DAYS = 7


def build_windows(daily_df: pd.DataFrame, window_days: int = WINDOW_DAYS) -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (windows, meta) where windows has shape [n_windows, window_days, 2*len(ALL_FEATURES)]
    and meta has one row per window: participant_id, window_start_date, window_end_date.
    """
    missing_cols = [f"missing_{f}" for f in ALL_FEATURES]
    windows = []
    meta_rows = []

    for participant_id, group in daily_df.groupby("participant_id"):
        group = group.sort_values("date").reset_index(drop=True)
        row_by_date = {row["date"]: i for i, row in group.iterrows()}
        sorted_dates = sorted(row_by_date.keys())

        for start_date in sorted_dates:
            start_ts = pd.Timestamp(start_date)
            expected_dates = [(start_ts + pd.Timedelta(days=i)).date().isoformat() for i in range(window_days)]
            if not all(d in row_by_date for d in expected_dates):
                continue

            day_rows = [group.iloc[row_by_date[d]] for d in expected_dates]
            feature_matrix = np.array(
                [[row[f] for f in ALL_FEATURES] for row in day_rows], dtype=np.float64
            )
            missing_matrix = np.array(
                [[bool(row[c]) for c in missing_cols] for row in day_rows], dtype=np.float64
            )
            # NaN feature values (where missing=True) are filled with 0.0; the
            # missing mask carries the "don't trust this value" signal instead,
            # so the model never sees a NaN, only a flagged placeholder.
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

            window = np.concatenate([feature_matrix, missing_matrix], axis=1)  # [window_days, 24]
            windows.append(window)
            meta_rows.append(
                {
                    "participant_id": participant_id,
                    "window_start_date": expected_dates[0],
                    "window_end_date": expected_dates[-1],
                }
            )

    if not windows:
        return np.empty((0, window_days, 2 * len(ALL_FEATURES))), pd.DataFrame(
            columns=["participant_id", "window_start_date", "window_end_date"]
        )
    return np.stack(windows), pd.DataFrame(meta_rows)


def main() -> int:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    daily_path = processed_dir / "daily_features.parquet"

    if not daily_path.exists():
        print(f"ERROR: {daily_path} not found. Run build_daily_features.py first.")
        return 1

    daily_df = pd.read_parquet(daily_path)
    windows, meta = build_windows(daily_df)

    out_windows = processed_dir / "sequences.npy"
    out_meta = processed_dir / "sequences_meta.parquet"
    np.save(out_windows, windows)
    meta.to_parquet(out_meta, index=False)

    print(f"Built {len(windows)} windows of shape {windows.shape[1:]} from {meta['participant_id'].nunique()} participants")
    print(f"Saved to {out_windows} and {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
