"""Timezone-aware day-boundary utilities shared by the feature pipeline.

StudentLife timestamps are Unix epoch seconds, documented as Eastern Time.
Per docs/PRODUCT_SPEC.md section 7.1, conversion must use the IANA zone
America/New_York (not a fixed UTC offset) so daylight-saving transitions are
handled correctly, and every feature row represents one local calendar day
00:00:00-23:59:59.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/New_York")


def to_local(unix_ts: int) -> datetime:
    """Convert a Unix epoch-seconds timestamp to a timezone-aware local datetime."""
    return datetime.fromtimestamp(unix_ts, tz=TZ)


def local_date(unix_ts: int) -> str:
    """The ISO local calendar date (YYYY-MM-DD) a timestamp falls on."""
    return to_local(unix_ts).date().isoformat()


def day_bounds(date_str: str) -> tuple[datetime, datetime]:
    """Timezone-aware [00:00:00, 23:59:59] bounds for a local calendar date."""
    start = datetime.fromisoformat(date_str).replace(tzinfo=TZ)
    end = start + timedelta(days=1) - timedelta(microseconds=1)
    return start, end


def split_interval_by_day(start_ts: int, end_ts: int) -> list[tuple[str, int, int]]:
    """Split a [start_ts, end_ts) Unix-second interval into per-local-day pieces.

    Returns a list of (date_str, overlap_seconds) is not quite right for callers
    that also need the clipped start/end (e.g. for night-overlap calculation), so
    this returns (date_str, clipped_start_ts, clipped_end_ts) for each local day
    the interval touches.
    """
    if end_ts <= start_ts:
        return []

    pieces = []
    cursor = start_ts
    while cursor < end_ts:
        date_str = local_date(cursor)
        day_start, day_end = day_bounds(date_str)
        day_end_ts = int(day_end.timestamp()) + 1  # exclusive upper bound, next midnight
        piece_end = min(end_ts, day_end_ts)
        pieces.append((date_str, cursor, piece_end))
        cursor = piece_end
    return pieces


def night_overlap_seconds(clipped_start_ts: int, clipped_end_ts: int, date_str: str) -> int:
    """Seconds of [clipped_start_ts, clipped_end_ts) that fall within 00:00-06:00
    local time on the given local calendar date."""
    day_start, _ = day_bounds(date_str)
    night_start_ts = int(day_start.timestamp())
    night_end_ts = night_start_ts + 6 * 3600
    overlap_start = max(clipped_start_ts, night_start_ts)
    overlap_end = min(clipped_end_ts, night_end_ts)
    return max(0, overlap_end - overlap_start)
