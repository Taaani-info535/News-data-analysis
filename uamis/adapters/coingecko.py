"""CoinGecko public /global endpoint (free, no key, rate-limited): BTC dominance and total market-cap 24h change.
Optional context for the crypto-market evidence; any failure just means that evidence is reported as unavailable."""
from __future__ import annotations
from typing import Dict, Optional


def parse_global(payload: dict) -> Optional[Dict[str, float]]:
    d = (payload or {}).get("data")
    if not isinstance(d, dict):
        return None
    try:
        dom = float(d["market_cap_percentage"]["btc"])
        chg = float(d["market_cap_change_percentage_24h_usd"])
    except (KeyError, TypeError, ValueError):
        return None
    out = {"btc_dominance": dom, "mcap_change_24h_pct": chg}
    try:
        out["total_mcap_usd"] = float(d["total_market_cap"]["usd"])
    except (KeyError, TypeError, ValueError):
        pass
    return out


def fetch_global(http, url: str) -> Optional[Dict[str, float]]:
    return parse_global(http.get_json(url))
