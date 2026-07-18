"""Parse StudentLife EMA_definition.json and per-participant EMA response files.

Per docs/PRODUCT_SPEC.md section 9: "Read the answer mappings from
dataset/EMA/EMA_definition.json. Do not hard-code assumed meanings for numeric
option indices." Verified against the real file: option codes are not a
monotonic scale in general — e.g. Stress's `level` question is
"[1]A little stressed, [2]Definitely stressed, [3]Stressed out, [4]Feeling good,
[5]Feeling great" — codes 4-5 are the *opposite* end (not stressed), not a
continuation of severity. Any evaluation code must decode through this mapping,
never assume "higher code = more stress."

Response files also contain a location field (sometimes correctly under the key
"location", sometimes under a malformed "null" key — both confirmed present in
the real Stress_u00.json) holding raw lat/long GPS coordinates. Per
docs/PRODUCT_SPEC.md section 3 (no precise GPS tracking), this is never parsed
into any feature — it is dropped unconditionally by this module.
"""

import json
import re
from pathlib import Path

import pandas as pd

from timeutils import local_date, to_local

_OPTION_PATTERN = re.compile(r"\[(\d+)\]([^,\[]+)")
_LOCATION_KEYS = {"location", "null"}


def parse_options(options_str: str) -> dict:
    """Parse an options string like '[1]A little stressed, [2]Definitely
    stressed, ' into {1: "A little stressed", 2: "Definitely stressed"}.
    Returns {} if the string doesn't use the [n]label format (e.g. Mood's
    "(Yes) 1 2 (No)" free-form questions) — such options are not silently
    guessed at.
    """
    matches = _OPTION_PATTERN.findall(options_str or "")
    return {int(code): label.strip() for code, label in matches}


def load_ema_definition(path: Path) -> dict:
    """Return {category_name: {question_id: {"text": str, "options": {code: label}}}}."""
    with open(path, "r", encoding="utf-8") as f:
        definition = json.load(f)

    categories = {}
    for entry in definition:
        name = entry.get("name", "<unnamed>")
        questions = {}
        for q in entry.get("questions", []):
            qid = q.get("question_id")
            if not qid:
                continue
            questions[qid] = {
                "text": q.get("question_text", ""),
                "options": parse_options(q.get("options", "")),
            }
        categories[name] = questions
    return categories


def parse_ema_response_file(path: Path, question_ids: list[str]) -> pd.DataFrame:
    """Parse one participant's EMA response JSON into a tidy DataFrame with
    columns [date, resp_time, <question_id> for each requested question_id].
    GPS/location fields are always dropped. Values are kept as the raw string
    codes from the file (decoding to labels is the caller's responsibility via
    load_ema_definition, kept separate so evaluation code must go through the
    real option mapping rather than a hard-coded one).
    """
    with open(path, "r", encoding="utf-8") as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError:
            return pd.DataFrame(columns=["date", "resp_time", *question_ids])

    rows = []
    for record in records:
        if "resp_time" not in record:
            continue
        resp_time = int(record["resp_time"])
        row = {"date": local_date(resp_time), "resp_time": resp_time}
        for qid in question_ids:
            if qid in _LOCATION_KEYS:
                continue
            row[qid] = record.get(qid)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["date", "resp_time", *question_ids])
    return pd.DataFrame(rows)
