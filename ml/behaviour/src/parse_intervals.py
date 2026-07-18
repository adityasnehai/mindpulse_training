"""Parse StudentLife sensing/phonelock and sensing/phonecharge interval files.

Both are structurally identical: two columns `start`, `end` (Unix seconds).
Verified against the real archive: StudentLife's released files already only
contain intervals lasting >= 60 minutes (docs/PRODUCT_SPEC.md section 6.1's
documented limitation — confirmed empirically: min observed duration in
phonelock_u00.csv is 60.27 minutes). This module does not re-apply a 60-minute
filter to the source rows for that reason, but the Android-side data (raw
keyguard/charging events, not pre-filtered) will need that filter applied
before features are comparable — that threshold lives in configs/base.yaml
(`features.long_interval_threshold_minutes`) and is applied in
build_daily_features.py so both sources go through the same rule explicitly,
rather than silently relying on the StudentLife file already being filtered.
"""

from pathlib import Path

import pandas as pd

from timeutils import night_overlap_seconds, split_interval_by_day


def merge_overlapping_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union overlapping/touching (start, end) intervals into non-overlapping spans.

    Required because the raw StudentLife phonelock/phonecharge logs contain
    genuinely overlapping intervals for the same participant (confirmed: e.g.
    u13 on 2013-04-28 has three logged lock intervals that overlap each other
    by tens of minutes — a logging artifact of the original passive-sensing
    collection, not a parsing bug). Summing overlapping raw intervals directly
    would double-count wall-clock time and can exceed 1440 minutes in a single
    calendar day, which is physically impossible — merging first is required
    for the total/longest/count features to be meaningful.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # overlaps or touches the previous merged span
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def parse_interval_file(csv_path: Path) -> pd.DataFrame:
    """Return a DataFrame of per-day-clipped interval segments:
    columns = [date, session_id, start_ts, end_ts, duration_seconds, night_overlap_seconds].
    Overlapping raw intervals are merged first (see merge_overlapping_intervals).
    `session_id` identifies one merged interval so multi-day sessions can be
    counted once per touched day but still traced back to one underlying event.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.sort_values("start").reset_index(drop=True)

    raw_intervals = [(int(r.start), int(r.end)) for r in df.itertuples(index=False) if r.end > r.start]
    merged_intervals = merge_overlapping_intervals(raw_intervals)

    segments = []
    for session_id, (start_ts, end_ts) in enumerate(merged_intervals):
        for date_str, clipped_start, clipped_end in split_interval_by_day(start_ts, end_ts):
            segments.append(
                {
                    "date": date_str,
                    "session_id": session_id,
                    "start_ts": clipped_start,
                    "end_ts": clipped_end,
                    "duration_seconds": clipped_end - clipped_start,
                    "night_overlap_seconds": night_overlap_seconds(clipped_start, clipped_end, date_str),
                }
            )

    if not segments:
        return pd.DataFrame(
            columns=["date", "session_id", "start_ts", "end_ts", "duration_seconds", "night_overlap_seconds"]
        )
    return pd.DataFrame(segments)


def daily_interval_features(
    segments: pd.DataFrame,
    prefix: str,
    long_threshold_minutes: int = 60,
    include_night_overlap: bool = False,
) -> pd.DataFrame:
    """Aggregate per-day-clipped segments into one row per date with:
    {prefix}_total_minutes, longest_{prefix}_minutes, {prefix}_session_count
    (only counting sessions whose ORIGINAL total duration, before day-clipping,
    is >= long_threshold_minutes), and optionally night_charge_overlap_minutes.
    """
    total_col = f"long_{prefix}_total_minutes"
    longest_col = f"longest_long_{prefix}_minutes"
    count_col = f"long_{prefix}_session_count"
    columns = ["date", total_col, longest_col, count_col]
    if include_night_overlap:
        columns.append("night_charge_overlap_minutes")

    if segments.empty:
        return pd.DataFrame(columns=columns)

    # Eligibility is evaluated on each session's ORIGINAL total duration (sum
    # across all day-pieces it was split into), not on any individual clipped
    # piece — a session split by midnight into two 40-minute pieces is still
    # one 80-minute long session, and both pieces count.
    session_totals = segments.groupby("session_id")["duration_seconds"].sum()
    long_session_ids = set(session_totals[session_totals >= long_threshold_minutes * 60].index)
    long_segments = segments[segments["session_id"].isin(long_session_ids)]

    rows = []
    for date_str, day_df in long_segments.groupby("date"):
        row = {
            "date": date_str,
            total_col: day_df["duration_seconds"].sum() / 60.0,
            longest_col: day_df["duration_seconds"].max() / 60.0,
            count_col: day_df["session_id"].nunique(),
        }
        if include_night_overlap:
            row["night_charge_overlap_minutes"] = day_df["night_overlap_seconds"].sum() / 60.0
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
