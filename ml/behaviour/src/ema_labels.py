"""Build within-person high-stress-day labels from real Stress EMA responses,
for evaluating anomaly-detection baselines against (docs/PRODUCT_SPEC.md
section 9).

The raw `level` code for the Stress category is NOT a monotonic severity scale
(verified against the real EMA_definition.json): codes 1-3 escalate in
stress ("a little stressed" -> "definitely stressed" -> "stressed out"), but
codes 4-5 are the *opposite* end of the same question ("feeling good",
"feeling great" — not stressed at all), not a continuation of severity. Using
the raw numeric code as an ordinal severity value would be wrong (it would
treat "feeling good" as more stressed than "stressed out").

SEVERITY_BY_LABEL below is a documented, considered re-ordering of the real
option *text* (not the numeric code) into an actual stress-severity scale:
higher = more stressed. This is an interpretive judgment call grounded in the
real label text ("stressed out" is unambiguously worse than "a little
stressed"; "feeling good"/"feeling great" are unambiguously the not-stressed
end) — it is not a fabricated mapping, but it is not handed to us verbatim
either, since the source file only provides display order, not severity
order. Any downstream evaluation code must go through this mapping (or
`load_ema_definition`'s real option text directly), never the raw code alone.
"""

from pathlib import Path

import pandas as pd

from parse_ema import load_ema_definition, parse_ema_response_file

SEVERITY_BY_LABEL = {
    "Stressed out": 4,
    "Definitely stressed": 3,
    "A little stressed": 2,
    "Feeling good": 1,
    "Feeling great": 0,
}


def stress_severity_map(ema_definition_path: Path) -> dict:
    """Return {code: severity} for the Stress/level question, built by joining
    the real option text against SEVERITY_BY_LABEL. Raises if the real file's
    label text doesn't match what we expect — fail loudly rather than
    silently mis-scoring severity if Dartmouth's wording differs from what
    was inspected."""
    categories = load_ema_definition(ema_definition_path)
    options = categories["Stress"]["level"]["options"]  # {code: label}
    severity_map = {}
    for code, label in options.items():
        if label not in SEVERITY_BY_LABEL:
            raise ValueError(
                f"Unrecognized Stress option label {label!r} (code {code}) — "
                "SEVERITY_BY_LABEL must be updated to match the real EMA_definition.json, "
                "not guessed at."
            )
        severity_map[code] = SEVERITY_BY_LABEL[label]
    return severity_map


def build_high_stress_labels(
    stress_response_dir: Path, ema_definition_path: Path, participant_ids: list[str]
) -> pd.DataFrame:
    """For each participant, decode their real Stress/level responses into a
    severity score, then label each response day as high-stress if its
    severity is in that PARTICIPANT'S OWN top quartile (within-person
    definition per docs/PRODUCT_SPEC.md section 9 — never a universal
    threshold). Returns columns [participant_id, date, severity, high_stress].
    Days with multiple responses use the max severity that day.
    """
    severity_map = stress_severity_map(ema_definition_path)
    rows = []
    for pid in participant_ids:
        response_path = stress_response_dir / f"Stress_{pid}.json"
        if not response_path.exists():
            continue
        df = parse_ema_response_file(response_path, question_ids=["level"])
        df = df.dropna(subset=["level"])
        if df.empty:
            continue
        df["severity"] = df["level"].astype(int).map(severity_map)
        daily_max = df.groupby("date")["severity"].max().reset_index()
        if len(daily_max) < 4:
            # Too few responses to define a meaningful within-person quartile.
            continue
        threshold = daily_max["severity"].quantile(0.75)
        daily_max["high_stress"] = daily_max["severity"] >= threshold
        daily_max["participant_id"] = pid
        rows.append(daily_max)

    if not rows:
        return pd.DataFrame(columns=["participant_id", "date", "severity", "high_stress"])
    return pd.concat(rows, ignore_index=True)
