"""Evidence builders. Each returns plain numbers/text computed from STORED data only; missing data => None (never invented)."""
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple

import numpy as np

from ..classify import hawkish_rule, surprise_sign
from ..features import hourly_vol
from ..models import Event
from ..prices import MarketData
from ..symbols import asset_kind

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
HAWK_WORDS = ("raise", "hike", "tighten", "restrictive", "hawkish", "higher for longer")
DOVE_WORDS = ("rate cut", "cuts rate", "lower rates", "easing", "ease ", "accommodat", "dovish", "weaker outlook", "downside risk")
SOURCE_CCY = {"federal reserve": "USD", "fomc": "USD", "ecb": "EUR", "european central bank": "EUR", "bank of england": "GBP",
              "bank of japan": "JPY", "swiss national bank": "CHF", "reserve bank of australia": "AUD", "bank of canada": "CAD"}


def decay(age_s: float, halflife_h: float) -> float:
    return math.exp(-math.log(2) * max(age_s, 0.0) / 3600.0 / max(halflife_h, 1e-6))


def market_open(now_ts: int, kind: str) -> bool:
    """Approximate FX/futures weekly closure (Fri 22:00 UTC -> Sun 21:00 UTC). Crypto is 24/7. Exchange holidays are NOT modelled."""
    if kind == "crypto":
        return True
    wd = ((now_ts // 86400) + 3) % 7          # 0=Mon .. 6=Sun (epoch day 0 was a Thursday)
    hr = (now_ts % 86400) / 3600.0
    if wd == 5:
        return False
    if wd == 6 and hr < 21:
        return False
    if wd == 4 and hr >= 22:
        return False
    return True


# ------------------------------------------------------------------------------------------------ macro surprises
def surprise_magnitude(store, ev: Event, min_hist: int) -> Tuple[float, str]:
    """(|strength| in 0..1, description). z-scores the surprise against this indicator's own past surprises when enough exist,
    otherwise falls back to sign-only at HALF strength (magnitude unknown)."""
    rows = store.db.execute("SELECT actual,forecast FROM events WHERE name=? AND currency=? AND t0<? AND actual IS NOT NULL AND forecast IS NOT NULL",
                            (ev.name, ev.currency, ev.t0)).fetchall()
    diffs = [r["actual"] - r["forecast"] for r in rows]
    if len(diffs) >= min_hist and float(np.std(diffs)) > 0:
        z = (ev.actual - ev.forecast) / float(np.std(diffs))
        if abs(z) > 8:      # a >8-sigma surprise is far more likely a unit/format mismatch in the feed than a real event: do not trust the size
            return 0.5, f"z-score implausible ({abs(z):.0f} sigma, check feed units); sign only"
        return math.tanh(abs(z) / 1.5), f"{abs(z):.1f} sigma vs {len(diffs)} past surprises"
    return 0.5, f"sign only (only {len(diffs)} past surprises; need {min_hist} to size it)"


def revision_note(store, ev: Event, hk: bool) -> Tuple[float, str]:
    """Detect a revision of the previous value from OUR OWN stored history (free feed gives no revision field)."""
    r = store.db.execute("SELECT actual FROM events WHERE name=? AND currency=? AND t0<? AND actual IS NOT NULL ORDER BY t0 DESC LIMIT 1",
                         (ev.name, ev.currency, ev.t0)).fetchone()
    if r is None or ev.previous is None or abs(ev.previous - r["actual"]) < 1e-9:
        return 0.0, ""
    # plausibility: a genuine revision is small relative to the values; a huge gap means unit/format mismatch between feed rows, not a revision
    if abs(ev.previous - r["actual"]) > 0.5 * max(abs(ev.previous), abs(r["actual"]), 0.5):
        return 0.0, ""
    up = ev.previous > r["actual"]
    return (0.25 if up == hk else -0.25), f"; previous revised {r['actual']:g}->{ev.previous:g}"


def macro_scores(store, cfg, now_ts: int) -> Dict[str, dict]:
    """Per currency: score in -1..+1 (positive = currency-supportive/hawkish) from RELEASED surprises with recency decay."""
    rc = cfg.raw["recommend"]
    lb, hl = int(rc["lookback_hours"]) * 3600, float(rc["halflife_hours"])
    cw = rc["category_weights"]
    out: Dict[str, dict] = {c: {"score": None, "items": [], "last_ts": None} for c in CURRENCIES}
    for ev in store.events_between(now_ts - lb, now_ts + 1, int(rc["min_event_importance"])):
        if ev.currency not in out or ev.actual is None or ev.forecast is None:
            continue
        hk = hawkish_rule(ev.category, ev.name)
        sg = surprise_sign(ev.actual, ev.forecast)
        if hk is None or not sg:
            continue
        hawkish = (sg > 0) == hk
        mag, mtext = surprise_magnitude(store, ev, int(rc["min_surprise_history"]))
        rev, rtext = revision_note(store, ev, hk)
        e = max(-1.0, min(1.0, (1 if hawkish else -1) * mag + rev))
        w = float(cw.get(ev.category, 0.25)) * ev.importance / 3.0 * decay(now_ts - ev.t0, hl)
        out[ev.currency]["items"].append((w * e, f"{ev.currency} {ev.name}: actual {ev.actual_raw} vs forecast {ev.forecast_raw or 'n/a'} "
                                                 f"-> {'hawkish' if hawkish else 'dovish'}/{'supports' if hawkish else 'weakens'} {ev.currency} ({mtext}{rtext})", ev.t0))
    for c, d in out.items():
        if d["items"]:
            d["score"] = math.tanh(sum(x[0] for x in d["items"]))
            d["items"].sort(key=lambda x: -abs(x[0]))
            d["last_ts"] = max(x[2] for x in d["items"])
    return out


# ------------------------------------------------------------------------------------------------ central-bank tone (coarse)
def tone_scores(store, cfg, now_ts: int) -> Dict[str, dict]:
    """Keyword tone of OFFICIAL central-bank headlines. Coarse by design: most titles are neutral, so this is often unavailable."""
    rc = cfg.raw["recommend"]
    lb, hl = int(rc["lookback_hours"]) * 3600, float(rc["halflife_hours"])
    out = {c: {"score": None, "items": []} for c in CURRENCIES}
    rows = store.db.execute("SELECT published,source,headline FROM news WHERE verified=1 AND published>=? AND published<=?", (now_ts - lb, now_ts + 300)).fetchall()
    for r in rows:
        src = r["source"].lower()
        ccy = next((c for k, c in SOURCE_CCY.items() if k in src), None)
        h = r["headline"].lower()
        hw, dv = any(w in h for w in HAWK_WORDS), any(w in h for w in DOVE_WORDS)
        if not ccy or hw == dv:
            continue
        out[ccy]["items"].append((0.5 * (1 if hw else -1) * decay(now_ts - r["published"], hl), f"{ccy} headline tone {'hawkish' if hw else 'dovish'} (keyword): {r['headline'][:70]}"))
    for c, d in out.items():
        if d["items"]:
            d["score"] = math.tanh(sum(x[0] for x in d["items"]))
    return out


# ------------------------------------------------------------------------------------------------ US yields (daily, FRED)
def rates_score(store, cfg, now_ts: int) -> Optional[dict]:
    """USD-supportive(+) if Treasury yields rose over the last 5 daily observations. Daily and delayed by nature."""
    rc = cfg.raw["recommend"]["yields"]
    chg, ts, used = [], None, []
    for name in ("US10Y", "US02Y"):
        rows = store.load_candles(name, 86400)
        if len(rows) < 6:
            continue
        if now_ts - (rows[-1]["ts"] + 86400) > int(rc["staleness_days"]) * 86400:
            continue
        chg.append(rows[-1]["close"] - rows[-6]["close"]); used.append(name); ts = max(ts or 0, rows[-1]["ts"])
    if not chg:
        return None
    d = float(np.mean(chg))
    return {"score": math.tanh(d / float(rc["change_scale_pp"])), "ts": ts,
            "text": f"{'/'.join(used)} {d:+.2f}pp over last 5 daily obs (daily, delayed data)"}


# ------------------------------------------------------------------------------------------------ price context
LOOKBACK = {900: (3600, 300), 3600: (4 * 3600, 3600), 14400: (24 * 3600, 3600), 86400: (5 * 86400, 3600)}   # horizon -> (lookback s, max candle res)


def price_z(md: Optional[MarketData], now_ts: int, horizon_s: int, labels_cfg: dict) -> Optional[Tuple[float, int]]:
    """Volatility-normalised trend over a horizon-appropriate lookback. None when data is missing/stale (weekend etc.)."""
    if md is None:
        return None
    lb, res = LOOKBACK[horizon_s]
    p1, p0 = md.price_at(now_ts, res), md.price_at(now_ts - lb, res)
    vol = hourly_vol(md, now_ts, int(labels_cfg.get("vol_lookback_hours", 24)), int(labels_cfg.get("min_vol_returns", 12)))
    if not (p1 and p0 and vol):
        return None
    return math.log(p1[0] / p0[0]) / (vol * math.sqrt(lb / 3600.0)), p1[1]


def currency_strength(pair_z: Dict[str, float], min_pairs: int = 12) -> Optional[Dict[str, float]]:
    """Currency-strength meter: solve z_pair ~= s_base - s_quote across all available forex pairs (least squares, sum(s)=0).
    Separates the COMMON currency component from pair-specific residuals."""
    pairs = {p: z for p, z in pair_z.items() if len(p) == 6 and p[:3] in CURRENCIES and p[3:] in CURRENCIES}
    if len(pairs) < min_pairs:
        return None
    idx = {c: i for i, c in enumerate(CURRENCIES)}
    A, y = [], []
    for p, z in pairs.items():
        row = np.zeros(len(CURRENCIES)); row[idx[p[:3]]] = 1; row[idx[p[3:]]] = -1
        A.append(row); y.append(z)
    A.append(np.ones(len(CURRENCIES)) * 10); y.append(0.0)
    s, *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    return {c: float(s[i]) for c, i in idx.items()}


def freshness(md: Optional[MarketData], now_ts: int, kind: str, cfg) -> Tuple[str, Optional[int], Optional[int]]:
    """(label, last_close_ts, age_s). RECENT <=30min, DELAYED <=max_staleness, STALE beyond, CLOSED on weekly closure.
    NOTE: provider-side delays (e.g. 15-minute delayed quotes) cannot be detected from candles alone."""
    last = md.last_close_ts(now_ts) if md else None
    if last is None:
        return "UNAVAILABLE", None, None
    age = now_ts - last[0]
    if not market_open(now_ts, kind):
        return "CLOSED", last[0], age
    max_st = int((cfg.g("candles", "max_staleness_seconds", default={}) or {}).get(3600, 7200))
    return ("RECENT" if age <= 1800 else "DELAYED" if age <= max_st else "STALE"), last[0], age


def risk_sentiment(cfg, mds: Dict[str, MarketData], now_ts: int) -> Optional[dict]:
    """Risk-on(+)/off(-) proxy: 24h volatility-normalised trend of the configured equity-index futures (fresh data only)."""
    vals, ts = [], None
    for a, m in cfg.all_assets().items():
        if asset_kind(m) != "index":
            continue
        r = price_z(mds.get(a), now_ts, 14400, cfg.raw["labels"])
        if r:
            vals.append(math.tanh(r[0] / 2)); ts = max(ts or 0, r[1])
    if not vals:
        return None
    return {"score": float(np.mean(vals)), "ts": ts, "n": len(vals), "text": f"equity-index futures 24h trend {float(np.mean(vals)):+.2f} (n={len(vals)})"}
