"""Time and statistics helpers. All internal times are UTC epoch seconds (int)."""
from __future__ import annotations
import math
import re
from datetime import datetime, timezone
from typing import Optional, Tuple

UTC = timezone.utc


def utcnow_ts() -> int:
    return int(datetime.now(tz=UTC).timestamp())


def ts_to_iso(ts: Optional[int]) -> str:
    if ts is None:
        return "n/a"
    return datetime.fromtimestamp(int(ts), tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def parse_iso_to_ts(s: str) -> int:
    """Parse ISO-8601 with offset (or naive = UTC) to epoch seconds."""
    if not s:
        raise ValueError("empty timestamp")
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


_NUM = re.compile(r"^\s*([<>~]?)\s*(-?\d+(?:[.,]\d+)?)\s*([%KkMmBbTt]?)\s*$")
_SCALE = {"": 1.0, "%": 1.0, "k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def parse_value(raw: Optional[str]) -> Optional[float]:
    """Parse '3.2%', '227K', '-0.1', '1.23M' into a float. Returns None if blank/unparseable.
    Never invents a value."""
    if raw is None:
        return None
    raw = str(raw).strip()
    if raw in ("", "-", "N/A", "n/a", "None"):
        return None
    m = _NUM.match(raw.replace(",", "."))
    if not m:
        return None
    return float(m.group(2)) * _SCALE[m.group(3).lower()]


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))
