"""OFFLINE DEMO on SYNTHETIC data. Prices and events are random numbers with a weak planted pattern so the pipeline has something to learn.
The output proves the software runs end-to-end; it says NOTHING about real-market performance."""
from __future__ import annotations
import os
import tempfile
from typing import List, Optional

import numpy as np

from . import pipeline
from .adapters.calendar_ff import make_event_id
from .classify import classify_event, hawkish_rule, surprise_rel, surprise_sign, theoretical_impact
from .config import Config, load_config
from .models import Event
from .storage import Store

BANNER = "*** SYNTHETIC DEMO DATA - random numbers, NOT market data. Results mean nothing about real performance. ***"
def _shipped_pairs() -> List[str]:
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "forex_pairs.txt")) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


DEMO_PAIRS = _shipped_pairs()          # the pair NAMES are yours; prices/events in the demo are still synthetic


def make_demo_config(tmp: str, pairs: Optional[List[str]] = None, days: int = 365, base_cfg: Optional[Config] = None) -> Config:
    cfg = base_cfg or load_config()
    raw = dict(cfg.raw)
    pf = os.path.join(tmp, "forex_pairs.txt")
    with open(pf, "w") as f:
        f.write("\n".join(pairs if pairs is not None else DEMO_PAIRS))
    raw["forex"] = dict(raw["forex"], pairs_file=pf)
    raw["db_path"] = os.path.join(tmp, "uamis.sqlite")
    raw["model_dir"] = os.path.join(tmp, "models")
    raw["assets"] = {k: v for k, v in raw["assets"].items() if k in ("XAUUSD", "DXY", "NAS100", "BTCUSD")}
    raw["ml"] = dict(raw["ml"], min_train_samples=150, min_test_samples=40)
    return Config(raw)


def synth_market(cfg: Config, store: Store, end_ts: int, days: int = 365, seed: int = 11, events: Optional[List[Event]] = None):
    rng = np.random.default_rng(seed)
    n5 = days * 288
    start = (end_ts // 300) * 300 - n5 * 300
    ts = start + np.arange(n5) * 300
    for a, meta in cfg.all_assets().items():
        vol5 = 0.0006 if meta.get("class") != "forex" else 0.00025
        if meta.get("base") in ("BTC",):
            vol5 = 0.0015
        r = rng.normal(0, vol5, n5)
        for ev in events or []:
            if ev.actual is None or ev.forecast is None:
                continue
            i = int((ev.t0 - start) // 300) + 1
            if 0 <= i < n5 - 3:
                th, _ = theoretical_impact(ev.currency, hawkish_rule(ev.category, ev.name), surprise_sign(ev.actual, ev.forecast), meta)
                sr = surprise_rel(ev.actual, ev.forecast, ev.previous) or 0.0
                eff = {"BULLISH": 1, "BEARISH": -1}.get(th, 0)
                r[i:i + 3] += eff * abs(sr) * vol5 * 6 + rng.normal(0, vol5 * 4, 3)   # weak planted effect + large noise
        px = 100 * np.exp(np.cumsum(r))
        o = np.concatenate([[100], px[:-1]]); h = np.maximum(o, px) * 1.0002; l = np.minimum(o, px) * 0.9998
        dow = ((ts // 86400) + 3) % 7          # 0=Mon (epoch day 0 was Thursday)
        keep = np.ones(n5, bool) if meta.get("base") in ("BTC", "ETH") else (dow < 5)
        # hourly aggregation from 5m (full hours only)
        nh = n5 // 12
        O, H, L, C = (o[:nh * 12].reshape(nh, 12)[:, 0], h[:nh * 12].reshape(nh, 12).max(1), l[:nh * 12].reshape(nh, 12).min(1), px[:nh * 12].reshape(nh, 12)[:, -1])
        hts = ts[:nh * 12:12]
        hk = keep[:nh * 12:12]
        store.upsert_candles(a, 3600, [(int(hts[i]), O[i], H[i], L[i], C[i], 1.0) for i in range(nh) if hk[i]])
        cut = n5 - 45 * 288
        store.upsert_candles(a, 300, [(int(ts[i]), o[i], h[i], l[i], px[i], 1.0) for i in range(cut, n5) if keep[i]])
    return start


def synth_events(end_ts: int, days: int = 365, seed: int = 5) -> List[Event]:
    rng = np.random.default_rng(seed)
    cats = [("CPI y/y", "USD"), ("Non-Farm Employment Change", "USD"), ("Federal Funds Rate", "USD"), ("Retail Sales m/m", "USD"),
            ("Core PCE Price Index m/m", "USD"), ("CPI y/y", "EUR"), ("Official Bank Rate", "GBP")]
    evs = []
    day0 = (end_ts // 86400 - days + 3) * 86400
    for d in range(0, days - 3, 2):
        t0 = day0 + d * 86400 + int(13.5 * 3600)
        if ((t0 // 86400) + 3) % 7 >= 5:
            continue
        name, cur = cats[int(rng.integers(len(cats)))]
        prev = round(float(rng.normal(3, 0.5)), 1); fc = round(prev + float(rng.normal(0, 0.1)), 1); act = round(fc + float(rng.normal(0, 0.15)), 1)
        cat, _ = classify_event(name)
        evs.append(Event(make_event_id(cur, name, t0), name, cur, t0, 3, cat, f"{fc}%", f"{prev}%", f"{act}%", fc, prev, act, "synthetic", t0, t0, t0))
    return evs


def run_demo(out=print) -> int:
    tmp = tempfile.mkdtemp(prefix="uamis_demo_")
    out(BANNER)
    cfg = make_demo_config(tmp)
    store = Store(cfg.db_path)
    end = 1_800_000_000 - (1_800_000_000 % 86400) + 10 * 3600     # arbitrary fixed 'now' for reproducibility
    events = synth_events(end - 86400)
    for e in events:
        store.upsert_event(e)
    synth_market(cfg, store, end, events=events)
    now = ((end // 300) * 300) + 60
    out("Training on synthetic history (a minute or so)...")
    reports = pipeline.train_all(cfg, store, now)
    for k, v in sorted(reports.items()):
        out(f"  {k}: {v['status']} samples={v['n_samples']} {v.get('reason', '')}")
    # live-like scenario: a pre event 30 min ahead and a just-released event
    pre = Event(make_event_id("USD", "CPI y/y", now + 1800), "CPI y/y", "USD", now + 1800, 3, "CPI", "3.1%", "3.0%", "", 3.1, 3.0, None, "demo", now, now, None)
    post_t0 = now - 1200          # released 20 min ago => the 15-minute post-news check is due in the same cycle
    post = Event(make_event_id("USD", "Non-Farm Employment Change", post_t0), "Non-Farm Employment Change", "USD", post_t0, 3, "NFP", "180K", "170K", "230K", 180e3, 170e3, 230e3, "demo", now, now, now)
    store.upsert_event(pre); store.upsert_event(post)
    out("\n===== tick (dry-run): heads-up alert + ONE consolidated market-direction report =====\n")
    print_fn = out
    res = pipeline.run_tick(cfg, store, None, None, now, dry_run=True, out=print_fn)
    out(f"tick result: {res}")
    out("\n===== evaluation =====")
    r = pipeline.run_evaluate(cfg, store, None, now, False, True, out)
    out(r["summary"])
    out(BANNER)
    store.close()
    return 0
