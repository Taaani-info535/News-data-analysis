"""Binance public market-data klines (no key). Base URL configurable; data-api.binance.vision serves public market data."""
from __future__ import annotations
from typing import List

BASE = "https://data-api.binance.vision/api/v3/klines"
INTERVAL = {60: "1m", 300: "5m", 900: "15m", 3600: "1h", 14400: "4h", 86400: "1d"}


def parse_klines(payload: list, res: int, now_ts: int) -> List[tuple]:
    if not isinstance(payload, list):
        raise ValueError("binance: unexpected payload")
    rows = []
    for k in payload:
        try:
            t = int(k[0]) // 1000
            if t + res > now_ts:
                continue
            rows.append((t, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
        except (IndexError, TypeError, ValueError):
            continue
    return rows


def fetch_candles(http, symbol: str, res: int, lookback_s: int, now_ts: int, base: str = BASE, max_pages: int = 20) -> List[tuple]:
    start = (now_ts - lookback_s) * 1000
    rows: List[tuple] = []
    for _ in range(max_pages):
        page = http.get_json(base, params={"symbol": symbol, "interval": INTERVAL[res], "startTime": start, "limit": 1000})
        got = parse_klines(page, res, now_ts)
        if not isinstance(page, list) or not page:
            break
        rows.extend(got)
        nxt = int(page[-1][0]) + res * 1000
        if nxt <= start or len(page) < 1000:
            break
        start = nxt
    return rows
