"""Yahoo Finance public chart endpoint (unofficial; personal use only per Yahoo terms - see docs/DATA_SOURCES.md)."""
from __future__ import annotations
from typing import List

BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
INTERVAL = {60: "1m", 300: "5m", 900: "15m", 3600: "1h", 86400: "1d"}


def parse_chart(payload: dict, res: int, now_ts: int) -> List[tuple]:
    """Return closed candles only (open_ts + res <= now for intraday). Raises ValueError on malformed/error payloads."""
    chart = (payload or {}).get("chart") or {}
    if chart.get("error"):
        raise ValueError(f"yahoo error: {chart['error'].get('code') if isinstance(chart['error'], dict) else chart['error']}")
    result = chart.get("result")
    if not result:
        raise ValueError("yahoo: empty result")
    r = result[0]
    ts = r.get("timestamp") or []
    q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    o, h, l, c, v = (q.get(k) or [] for k in ("open", "high", "low", "close", "volume"))
    rows = []
    for i, t in enumerate(ts):
        try:
            vals = (o[i], h[i], l[i], c[i])
        except IndexError:
            continue
        if any(x is None for x in vals):
            continue          # gaps are left as gaps, never interpolated
        if res < 86400 and t + res > now_ts:
            continue          # still-forming candle
        rows.append((int(t), vals[0], vals[1], vals[2], vals[3], (v[i] if i < len(v) else 0) or 0))
    return rows


def fetch_candles(http, symbol: str, res: int, range_: str, now_ts: int) -> List[tuple]:
    url = BASE.format(symbol=symbol)
    data = http.get_json(url, params={"interval": INTERVAL[res], "range": range_, "includePrePost": "false"})
    return parse_chart(data, res, now_ts)
