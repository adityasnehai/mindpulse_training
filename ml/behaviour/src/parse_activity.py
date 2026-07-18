"""Parse StudentLife sensing/activity/activity_uXX.csv files.

Real file format (verified against the extracted archive): two columns,
`timestamp` (Unix seconds) and ` activity inference` (note leading space —
an artifact of the original CSV export, handled here rather than assumed away
silently). Values are near-continuous samples (roughly every few seconds while
the phone is sensing), not just transition events:

    0 = stationary, 1 = walking, 2 = running, 3 = unknown/other

Design decision (not specified verbatim in the product spec, which only names
the feature and its source column): a sample's activity value is attributed to
the wall-clock gap between it and the *next* sample. A gap longer than
`max_gap_seconds` is treated as a sensing outage (no attribution to any
activity, reduces coverage) rather than assumed to be continuously stationary —
otherwise a multi-hour phone-off period would be silently counted as
"stationary time," which would be fabricating data the sensor never observed.
"""

from pathlib import Path

import pandas as pd

from timeutils import local_date, split_interval_by_day

ACTIVITY_LABELS = {0: "stationary", 1: "walking", 2: "running", 3: "unknown"}
DEFAULT_MAX_GAP_SECONDS = 300  # 5 minutes; longer gaps are an outage, not "stationary"


def parse_activity_file(csv_path: Path, max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS) -> pd.DataFrame:
    """Return a DataFrame of per-day-clipped activity segments:
    columns = [date, label, start_ts, end_ts, duration_seconds].
    Segments crossing midnight are split so every row belongs to exactly one
    local calendar day (America/New_York).
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"activity inference": "activity"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    segments = []
    timestamps = df["timestamp"].to_numpy()
    activities = df["activity"].to_numpy()

    for i in range(len(df) - 1):
        start_ts = int(timestamps[i])
        end_ts = int(timestamps[i + 1])
        gap = end_ts - start_ts
        if gap <= 0 or gap > max_gap_seconds:
            continue
        label = ACTIVITY_LABELS.get(int(activities[i]), "unknown")
        for date_str, clipped_start, clipped_end in split_interval_by_day(start_ts, end_ts):
            segments.append(
                {
                    "date": date_str,
                    "label": label,
                    "start_ts": clipped_start,
                    "end_ts": clipped_end,
                    "duration_seconds": clipped_end - clipped_start,
                }
            )

    if not segments:
        return pd.DataFrame(columns=["date", "label", "start_ts", "end_ts", "duration_seconds"])
    return pd.DataFrame(segments)


def daily_activity_fractions(segments: pd.DataFrame, day_length_seconds: int = 86400) -> pd.DataFrame:
    """Aggregate per-day-clipped segments into one row per date with:
    stationary_fraction, walking_fraction, running_fraction, unknown_activity_fraction
    (fractions of *covered* time) and activity_coverage_fraction (covered / day length).
    """
    if segments.empty:
        return pd.DataFrame(
            columns=[
                "date", "stationary_fraction", "walking_fraction", "running_fraction",
                "unknown_activity_fraction", "activity_coverage_fraction",
            ]
        )

    rows = []
    for date_str, day_df in segments.groupby("date"):
        covered = day_df["duration_seconds"].sum()
        by_label = day_df.groupby("label")["duration_seconds"].sum()
        row = {"date": date_str}
        for label in ("stationary", "walking", "running", "unknown"):
            seconds = by_label.get(label, 0)
            row[f"{label}_fraction" if label != "unknown" else "unknown_activity_fraction"] = (
                seconds / covered if covered > 0 else None
            )
        row["activity_coverage_fraction"] = min(1.0, covered / day_length_seconds)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
