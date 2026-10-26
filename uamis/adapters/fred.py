"""Optional FRED adapter for Treasury yields (free API key in env FRED_API_KEY). Context only; not an ML feature."""
from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import List

URL = "https://api.stlouisfed.org/fred/series/observations"
SERIES = {"US10Y": "DGS10", "US02Y": "DGS2"}


def parse_observations(payload: dict) -> List[tuple]:
    rows = []
    for o in (payload or {}).get("observations", []):
        try:
            v = float(o["value"])           # '.' means missing -> ValueError -> skipped
            ts = int(datetime.strptime(o["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        except (KeyError, ValueError):
            continue
        rows.append((ts, v, v, v, v, 0.0))
    return rows


def fetch_series(http, series_id: str, start: str) -> List[tuple]:
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        raise RuntimeError("FRED_API_KEY not set")
    data = http.get_json(URL, params={"series_id": series_id, "api_key": key, "file_type": "json", "observation_start": start})
    return parse_observations(data)
