"""Data validation: candles, timestamps, staleness. Invalid data is dropped and counted, never repaired by invention."""
from __future__ import annotations
import math
from typing import Iterable, List, Optional, Tuple


def validate_candles(rows: Iterable[tuple], res: int, now_ts: int, max_future_s: int = 300) -> Tuple[List[tuple], int]:
    """rows: (ts, o, h, l, c, v). Returns (clean_rows, n_rejected). Rejects NaN/None/non-positive prices, high<low,
    OHLC inconsistency, future timestamps, duplicate timestamps (keeps last)."""
    seen = {}
    rejected = 0
    for r in rows:
        try:
            ts, o, h, l, c = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])
            v = float(r[5]) if r[5] is not None and not (isinstance(r[5], float) and math.isnan(r[5])) else 0.0
        except (TypeError, ValueError):
            rejected += 1
            continue
        if any(math.isnan(x) or x <= 0 for x in (o, h, l, c)) or h < l or h < max(o, c) - 1e-12 * h or l > min(o, c) + 1e-12 * l:
            rejected += 1
            continue
        if ts > now_ts + max_future_s:
            rejected += 1
            continue
        seen[ts] = (ts, o, h, l, c, v)
    return [seen[k] for k in sorted(seen)], rejected


def staleness_seconds(last_ts: Optional[int], res: int, now_ts: int) -> Optional[int]:
    """Age of the latest *closed* candle's close time relative to now. None if no data."""
    if last_ts is None:
        return None
    return max(0, now_ts - (last_ts + res))
