"""Point-in-time feature engineering and label construction.
Pre stage : features at feat_ts = t0 - pre_lead (everything known >= pre_lead before release); outcome window t0 -> t0+h.
Post stage: features at feat_ts = anchor = t0 + delay incl. surprise and early reaction; outcome window anchor -> anchor+h.
Labels: log return r over the outcome window; threshold thr = max(k * sigma_h, cost) with sigma_h = pre-feature hourly vol * sqrt(h/1h).
UP if r > thr, DOWN if r < -thr, else SIDEWAYS. thr is frozen into the prediction at creation time."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .classify import LEARNABLE_CATEGORIES, relevant_assets, surprise_rel, surprise_sign
from .models import Event
from .prices import MarketData

BASE_COLS = ["z1", "z4", "z24", "vol_pct", "vol_ratio", "dd24", "hour_sin", "hour_cos", "importance", "fcst_vs_prev", "has_forecast"]
POST_COLS = ["surprise_rel", "surprise_sign", "early_z", "has_actual"]


def feature_columns(stage: str, assets: List[str]) -> List[str]:
    cols = list(BASE_COLS) + (POST_COLS if stage == "post" else [])
    return cols + [f"cat_{c}" for c in LEARNABLE_CATEGORIES] + [f"asset_{a}" for a in assets]


@dataclass
class Sample:
    event_id: str
    asset: str
    stage: str
    horizon_s: int
    feat_ts: int
    anchor_ts: int
    end_ts: int
    category: str
    features: Dict[str, float]
    thr: float
    ret: Optional[float] = None
    label: Optional[str] = None


def label_from_return(ret: float, thr: float) -> str:
    return "UP" if ret > thr else ("DOWN" if ret < -thr else "SIDEWAYS")


def hourly_vol(md: MarketData, t: int, lookback: int, min_n: int) -> Optional[float]:
    s = md.hourly()
    if s is None:
        return None
    i = s.idx_at(t)
    if i < min_n:
        return None
    v = s.vol_roll(lookback, min_n)[i]
    if not np.isfinite(v) or v <= 0:
        return None
    # staleness: latest hourly candle must be recent
    if t - int(s.close_ts[i]) > 4 * 3600:
        return None
    return float(v)


def vol_percentile(md: MarketData, t: int, lookback: int, min_n: int, hist_bars: int = 2160) -> Optional[float]:
    s = md.hourly()
    if s is None:
        return None
    i = s.idx_at(t)
    if i < min_n:
        return None
    v = s.vol_roll(lookback, min_n)
    cur = v[i]
    hist = v[max(0, i - hist_bars):i]          # strictly earlier values only
    hist = hist[np.isfinite(hist)]
    if not np.isfinite(cur) or len(hist) < 50:
        return None
    return float((hist <= cur).mean())


def build_features(md: MarketData, ev: Event, stage: str, feat_ts: int, cfg_labels: dict, vol_cache: Optional[dict] = None) -> Optional[Dict[str, float]]:
    """Return numeric base features or None if any required data is missing (caller must skip - never impute)."""
    lb, mn = int(cfg_labels.get("vol_lookback_hours", 24)), int(cfg_labels.get("min_vol_returns", 12))
    vol = hourly_vol(md, feat_ts, lb, mn)
    pct = vol_percentile(md, feat_ts, lb, mn)
    if vol is None or pct is None:
        return None
    p0 = md.price_at(feat_ts, 3600)
    p1 = md.price_at(feat_ts - 3600, 3600)
    p4 = md.price_at(feat_ts - 4 * 3600, 3600)
    p24 = md.price_at(feat_ts - 24 * 3600, 3600)
    if not (p0 and p1 and p4 and p24):
        return None
    z1 = math.log(p0[0] / p1[0]) / vol
    z4 = math.log(p0[0] / p4[0]) / (vol * 2.0)
    z24 = math.log(p0[0] / p24[0]) / (vol * math.sqrt(24))
    s = md.hourly()
    i = s.idx_at(feat_ts)
    win = s.c[max(0, i - 23): i + 1]
    dd24 = (win[-1] / win.max() - 1.0) / (vol * math.sqrt(24)) if len(win) >= 12 else None
    r6 = s.hourly_returns()[max(0, i - 5): i + 1]
    r6 = r6[np.isfinite(r6)]
    vol_ratio = float(np.std(r6, ddof=1) / vol) if len(r6) >= 4 else None
    if dd24 is None or vol_ratio is None:
        return None
    hr = ((ev.t0 % 86400) / 3600.0)
    sc = max(abs(ev.forecast) if ev.forecast is not None else 0.0, abs(ev.previous) if ev.previous is not None else 0.0, 1e-9)
    has_f = ev.forecast is not None
    f_vs_p = ((ev.forecast - ev.previous) / sc) if (has_f and ev.previous is not None) else 0.0
    f = {"z1": z1, "z4": z4, "z24": z24, "vol_pct": pct, "vol_ratio": vol_ratio, "dd24": dd24,
         "hour_sin": math.sin(2 * math.pi * hr / 24), "hour_cos": math.cos(2 * math.pi * hr / 24),
         "importance": float(ev.importance), "fcst_vs_prev": max(-1.0, min(1.0, f_vs_p)), "has_forecast": 1.0 if has_f else 0.0,
         "_vol": vol}
    if stage == "post":
        pt0 = md.price_at(ev.t0, 300)
        pa = md.price_at(feat_ts, 300)
        if not (pt0 and pa) or ev.actual is None:
            return None
        sr = surprise_rel(ev.actual, ev.forecast, ev.previous)
        ss = surprise_sign(ev.actual, ev.forecast)
        elapsed = max(feat_ts - ev.t0, 60)
        f.update({"surprise_rel": sr if sr is not None else 0.0, "surprise_sign": float(ss) if ss is not None else 0.0,
                  "early_z": math.log(pa[0] / pt0[0]) / (vol * math.sqrt(elapsed / 3600.0)), "has_actual": 1.0 if sr is not None else 0.0})
    return f


def anchors(ev: Event, stage: str, pre_lead_s: int, post_delay_s: int):
    """(feat_ts, anchor_ts)"""
    if stage == "pre":
        return ev.t0 - pre_lead_s, ev.t0
    # Match live post-news timing: do not forecast until the actual has been observed,
    # and never earlier than the configured minimum post-release delay.
    seen = ev.actual_seen_at if ev.actual_seen_at is not None else ev.t0 + post_delay_s
    a = max((int(seen) // 300) * 300, ev.t0 + post_delay_s)
    return a, a


def one_hot(sample_feats: Dict[str, float], category: str, asset: str, assets: List[str]) -> Dict[str, float]:
    d = dict(sample_feats)
    for c in LEARNABLE_CATEGORIES:
        d[f"cat_{c}"] = 1.0 if c == category else 0.0
    for a in assets:
        d[f"asset_{a}"] = 1.0 if a == asset else 0.0
    return d


def make_sample(md: MarketData, ev: Event, asset: str, stage: str, horizon_s: int, cfg, feat_ts: Optional[int] = None,
                anchor_ts: Optional[int] = None, compute_outcome: bool = True, assets: Optional[List[str]] = None) -> Optional[Sample]:
    labels = cfg.raw["labels"]
    pre_lead = int(cfg.g("calendar", "pre_lead_minutes", default=60)) * 60
    post_delay = int(cfg.g("calendar", "post_delay_minutes", default=15)) * 60
    if feat_ts is None or anchor_ts is None:
        feat_ts, anchor_ts = anchors(ev, stage, pre_lead, post_delay)
    base = build_features(md, ev, stage, feat_ts, labels)
    if base is None:
        return None
    vol = base.pop("_vol")
    cost = float(cfg.all_assets()[asset].get("cost_bps", 2)) / 1e4 if asset in cfg.all_assets() else 2e-4
    thr = max(float(labels["sideways_k"]) * vol * math.sqrt(horizon_s / 3600.0), cost)
    feats = one_hot(base, ev.category, asset, assets or sorted(cfg.all_assets()))
    end_ts = anchor_ts + horizon_s
    smp = Sample(ev.event_id, asset, stage, horizon_s, feat_ts, anchor_ts, end_ts, ev.category, feats, thr)
    if compute_outcome:
        max_res = max(300, horizon_s // 3)
        pa = md.price_at(anchor_ts, max_res)
        pe = md.price_at(end_ts, max_res)
        if pa and pe:
            smp.ret = math.log(pe[0] / pa[0])
            smp.label = label_from_return(smp.ret, thr)
    return smp


def build_samples(events: List[Event], mds: Dict[str, MarketData], stage: str, horizon_s: int, cfg, now_ts: int) -> List[Sample]:
    """Historical, *labelled* samples only; outcome windows must have completed before now."""
    assets_all = cfg.all_assets()
    names = sorted(assets_all)
    out: List[Sample] = []
    min_imp = int(cfg.g("calendar", "min_importance", default=2))
    for ev in events:
        if ev.importance < min_imp or ev.category not in LEARNABLE_CATEGORIES:
            continue
        if stage == "post" and (ev.actual is None or ev.forecast is None):
            continue
        for a in relevant_assets_cfg(cfg, ev.currency):
            md = mds.get(a)
            if md is None:
                continue
            s = make_sample(md, ev, a, stage, horizon_s, cfg, assets=names)
            if s and s.label is not None and s.end_ts <= now_ts:
                out.append(s)
    out.sort(key=lambda s: (s.anchor_ts, s.asset))
    return out


def relevant_assets_cfg(cfg, currency: str) -> List[str]:
    assets = cfg.all_assets()
    if cfg.g("relevance_mode", default="strict") == "all":
        return sorted(assets)
    return relevant_assets(currency, assets)
