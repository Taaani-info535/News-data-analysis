"""Combine calibrated ML probabilities, historical event study and the risk filter into a ledger-ready Prediction."""
from __future__ import annotations
import hashlib
import json
from typing import Dict, List, Optional

from . import eventstudy, ml
from .classify import hawkish_rule, theoretical_impact
from .features import Sample, make_sample
from .models import LABELS, Event, Prediction
from .prices import MarketData
from .risk import RiskInput, assess

DRIVER_LABEL = {"z1": "last-1h move", "z4": "last-4h move", "z24": "last-24h move", "vol_pct": "volatility percentile",
                "vol_ratio": "recent/24h vol ratio", "dd24": "drawdown from 24h high", "surprise_rel": "surprise size",
                "surprise_sign": "surprise direction", "early_z": "initial reaction", "fcst_vs_prev": "forecast vs previous",
                "importance": "event importance", "hour_sin": "time of day", "hour_cos": "time of day", "has_forecast": "forecast availability"}


def pred_id(event_id: str, asset: str, stage: str, horizon_s: int) -> str:
    return hashlib.sha1(f"{event_id}|{asset}|{stage}|{horizon_s}".encode()).hexdigest()[:16]


def _top(p: Dict[str, float]):
    if not p:
        return None, None
    k = max(p, key=p.get)
    return k, p[k]


def predict_one(cfg, store_hist: Dict, ev: Event, asset: str, stage: str, horizon_s: int, md: MarketData, now_ts: int,
                hist_samples: List[Sample], deteriorated: bool = False, other_event_minutes: Optional[float] = None,
                feat_ts: Optional[int] = None, anchor_ts: Optional[int] = None) -> Prediction:
    """store_hist unused placeholder kept for signature stability. Returns a Prediction even when data is missing
    (status INSUFFICIENT DATA, decision STAY AWAY) so the ledger and reports are complete and honest."""
    names = sorted(cfg.all_assets())
    smp = make_sample(md, ev, asset, stage, horizon_s, cfg, feat_ts=feat_ts, anchor_ts=anchor_ts, compute_outcome=False, assets=names)
    pid = pred_id(ev.event_id, asset, stage, horizon_s)
    rc = cfg.raw["risk"]
    art = ml.load_artifact(cfg.model_dir, stage, horizon_s)
    version = art["version"] if art else "none"
    if smp is None:
        ri = RiskInput(stage, ev.importance, "INSUFFICIENT DATA", {}, data_missing=True)
        lvl, dec, why = assess(ri, rc)
        return Prediction(pid, ev.event_id, asset, stage, horizon_s, now_ts, anchor_ts or ev.t0, {}, {}, "UNCERTAIN",
                          "INSUFFICIENT DATA", version, 0, {"status": "MISSING MARKET DATA"}, lvl, dec, why, {}, None, None, ev.category)
    feats = smp.features
    ss = eventstudy.study(feats, ev.category, asset, stage, horizon_s, smp.feat_ts, hist_samples, int(cfg.g("study", "min_samples", default=20)),
                          int(cfg.g("study", "neighbors", default=40)), float(cfg.g("study", "confidence_z", default=1.96)), ev.event_id)
    ml_p = None
    if art is not None:
        ml_p = dict(zip(LABELS, ml.predict_proba(art, feats).tolist()))
    st_p = ss["freq"] if ss["status"] == "OK" else None
    baseline = dict(zip(LABELS, art["prior"])) if art else (st_p or {})
    ml_status = art["report"]["status"] if art else None

    if ml_p is not None and ml_status == "VALIDATED":
        probs, label = ml_p, "VALIDATED"
    elif ml_p is not None:
        probs, label = ml_p, "EXPERIMENTAL"
    elif st_p is not None:
        probs, label = st_p, "EXPERIMENTAL"
    else:
        probs, label = {}, "INSUFFICIENT DATA"
    if label != "INSUFFICIENT DATA" and ss["status"] != "OK" and label == "EXPERIMENTAL":
        label = "LOW CONFIDENCE"
    arg, topp = _top(probs)
    sarg, stop = _top(st_p) if st_p else (None, None)
    marg, mtop = _top(ml_p) if ml_p else (None, None)
    last = md.last_close_ts(now_ts)
    stale = (now_ts - last[0]) if last else None
    max_st = int((cfg.g("candles", "max_staleness_seconds", default={}) or {}).get(3600, 7200))
    ri = RiskInput(stage, ev.importance, label, probs, marg, mtop, sarg, stop, feats.get("vol_pct"), stale, max_st,
                   other_event_minutes, deteriorated,
                   late_fraction=max(0.0, (now_ts - smp.anchor_ts) / horizon_s) if stage == "post" else 0.0, released=(stage == "post"))
    lvl, dec, why = assess(ri, rc)
    min_top = float(rc["min_top_probability"])
    argmax = arg if (arg and topp >= min_top) else "UNCERTAIN"
    drivers: List[str] = []
    if art and art.get("importance"):
        for c, v in sorted(art["importance"].items(), key=lambda kv: -kv[1]):
            if c in DRIVER_LABEL and v > 0:
                drivers.append(f"{DRIVER_LABEL[c]} ({feats.get(c, 0):+.2f})")
            if len(drivers) == 3:
                break
    hist = {"status": ss["status"], "n": ss["n"], "n_same_category": ss["n_same_category"], "freq": ss.get("freq", {}),
            "ci": {k: [round(v[0], 3), round(v[1], 3)] for k, v in ss.get("ci", {}).items()}, "note": ss["note"],
            "ml_status": ml_status, "ml_test_brier_skill": (art["report"].get("test", {}).get("brier_skill") if art else None)}
    return Prediction(pid, ev.event_id, asset, stage, horizon_s, now_ts, smp.anchor_ts, probs, baseline, argmax, label, version,
                      ss["n"], hist, lvl, dec, why, {k: float(v) for k, v in feats.items() if not k.startswith(("cat_", "asset_"))},
                      smp.thr, feats.get("vol_pct"), ev.category, drivers)


def scenario_lines(cfg, ev: Event, asset: str, hist_post: List[Sample], horizon_s: int, now_ts: int) -> List[str]:
    """Pre-news conditional scenarios: textbook theory + measured history conditioned on surprise direction."""
    meta = cfg.all_assets()[asset]
    hk = hawkish_rule(ev.category, ev.name)
    lines = []
    for sign, word in ((1, "ABOVE forecast"), (-1, "BELOW forecast")):
        th, why = theoretical_impact(ev.currency, hk, sign, meta)
        st = eventstudy.scenario_stats(hist_post, asset, ev.category, horizon_s, sign, now_ts, int(cfg.g("study", "min_samples", default=20)))
        if st["status"] == "OK":
            f = st["freq"]
            hs = f"history n={st['n']}: UP {f['UP']:.0%} / DOWN {f['DOWN']:.0%} / SIDE {f['SIDEWAYS']:.0%}"
        else:
            hs = f"history: INSUFFICIENT HISTORICAL DATA (n={st['n']})"
        lines.append(f"If {word}: theory {th} ({why}); {hs}")
    return lines


def prediction_from_row(r) -> Prediction:
    return Prediction(r["pred_id"], r["event_id"], r["asset"], r["stage"], r["horizon_s"], r["created_ts"], r["anchor_ts"],
                      json.loads(r["probs"]), json.loads(r["baseline"]), r["argmax"], r["status_label"], r["model_version"],
                      r["hist_n"], json.loads(r["hist"]), r["risk_level"], r["decision"], json.loads(r["reasons"]),
                      json.loads(r["features"]), r["thr"], r["vol_pct"], r["category"] or "")
