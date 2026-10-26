"""Crypto market-wide context: breadth across liquid coins, total market-cap change and BTC dominance (CoinGecko, optional).
Distinguishes broad-market bias from coin-specific price behaviour. Crypto-specific NEWS is not covered (no reliable free source)."""
from __future__ import annotations
import json
import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

from ..adapters import coingecko
from ..symbols import asset_kind
from .evidence import price_z

log = logging.getLogger("uamis.recommend.crypto")
KV = "cg_dom_hist"


def fetch_global(cfg, store, http, now_ts: int) -> Optional[Dict[str, float]]:
    """One CoinGecko call per cycle; failure => None (evidence reported unavailable). Snapshots dominance to compute its own 24h change."""
    c = cfg.raw["recommend"]["crypto"]
    if http is None or not c.get("coingecko"):
        return None
    try:
        g = coingecko.fetch_global(http, c["coingecko_url"])
    except Exception as e:
        log.warning("coingecko unavailable: %s", type(e).__name__)
        return None
    if g:
        hist = json.loads(store.kv_get(KV, "[]") or "[]")
        if not hist or now_ts - hist[-1][0] >= 1800:
            hist.append([now_ts, g["btc_dominance"]])
            store.kv_set(KV, json.dumps(hist[-300:]))
        g["ts"] = now_ts
    return g


def dominance_delta(store, now_ts: int, dom_now: float) -> Optional[float]:
    hist = json.loads(store.kv_get(KV, "[]") or "[]")
    cands = [(abs((now_ts - t) - 86400), d) for t, d in hist if 20 * 3600 <= now_ts - t <= 28 * 3600]
    return dom_now - min(cands)[1] if cands else None


class CryptoContext:
    def __init__(self, cfg, store, mds, now_ts: int, global_data: Optional[Dict[str, float]]):
        self.cfg, self.now = cfg, now_ts
        self.coins = [a for a, m in cfg.all_assets().items() if asset_kind(m) == "crypto"]
        self.g = global_data
        self.dom_delta = dominance_delta(store, now_ts, global_data["btc_dominance"]) if global_data else None
        self.z: Dict[int, Dict[str, float]] = {}
        for h in cfg.horizons:
            hs = h * 60
            zs = {}
            for a in self.coins:
                r = price_z(mds.get(a), now_ts, hs, cfg.raw["labels"])
                if r:
                    zs[a] = r[0]
            self.z[hs] = zs

    def market(self, horizon_s: int) -> Tuple[Optional[float], str, Optional[int]]:
        zs = self.z.get(horizon_s, {})
        parts, bits = [], []
        if len(zs) >= 2:
            b = float(np.mean([math.tanh(z / 2) for z in zs.values()]))
            parts.append((0.6, b)); bits.append(f"breadth {sum(1 for z in zs.values() if z > 0)}/{len(zs)} coins up")
        if self.g:
            parts.append((0.4, math.tanh(self.g["mcap_change_24h_pct"] / 3.0)))
            bits.append(f"total mcap 24h {self.g['mcap_change_24h_pct']:+.1f}%")
            bits.append(f"BTC.D {self.g['btc_dominance']:.1f}%" + (f" ({self.dom_delta:+.2f}pp/24h)" if self.dom_delta is not None else ""))
        if not parts:
            return None, "crypto market breadth/market-cap unavailable", None
        w = sum(p[0] for p in parts)
        return sum(p[0] * p[1] for p in parts) / w, "crypto market: " + ", ".join(bits), self.now

    def alt_adjust(self, base: str) -> Tuple[float, str]:
        """Rising BTC dominance is relative pressure on non-BTC coins (coarse, small effect)."""
        if base == "BTC" or self.dom_delta is None:
            return 0.0, ""
        scale = float(self.cfg.raw["recommend"]["crypto"]["dominance_scale_pp"])
        return -0.2 * math.tanh(self.dom_delta / scale), f"BTC dominance {self.dom_delta:+.2f}pp/24h"
