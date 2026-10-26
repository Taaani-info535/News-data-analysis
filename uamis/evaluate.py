"""Outcome scoring and honest performance metrics. Pending predictions are never counted as right or wrong."""
from __future__ import annotations
import json
import math
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .features import label_from_return
from .models import LABELS
from .prices import MarketData
from .util import wilson_interval

NO_DATA_DEADLINE_S = 7 * 86400


def score_pending(store, cfg, now_ts: int) -> Dict[str, int]:
    """Score predictions whose outcome window (plus grace) has completed. Uses the thr frozen at prediction time."""
    grace = int(cfg.g("evaluation", "grace_seconds", default=600))
    hv = float(cfg.g("evaluation", "high_vol_pct", default=0.8))
    mds: Dict[str, MarketData] = {}
    res = {"evaluated": 0, "pending": 0, "no_data": 0}
    for r in store.pending_predictions():
        end = r["anchor_ts"] + r["horizon_s"]
        if now_ts < end + grace:
            res["pending"] += 1
            continue
        a = r["asset"]
        if a not in mds:
            mds[a] = MarketData.from_store(store, a)
        max_res = max(300, r["horizon_s"] // 3)
        pa, pe = mds[a].price_at(r["anchor_ts"], max_res), mds[a].price_at(end, max_res)
        if r["thr"] is None or not (pa and pe):
            if now_ts > end + NO_DATA_DEADLINE_S:
                store.set_outcome(r["pred_id"], None, None, "NO_DATA", now_ts, 0); res["no_data"] += 1
            else:
                res["pending"] += 1
            continue
        ret = math.log(pe[0] / pa[0])
        store.set_outcome(r["pred_id"], ret, label_from_return(ret, r["thr"]), "EVALUATED", now_ts,
                          int((r["vol_pct"] or 0) >= hv))
        res["evaluated"] += 1
    return res


def _group_metrics(rows: List[dict], cost_bps_map: Dict[str, float]) -> Dict:
    n = len(rows)
    out = {"n": n, "n_events": len({r["event_id"] for r in rows})}
    if n == 0:
        return out
    dec = [r for r in rows if r["argmax"] in ("UP", "DOWN")]
    out["coverage_directional"] = len(dec) / n
    if dec:
        k = sum(1 for r in dec if r["argmax"] == r["outcome_label"])
        out["directional_accuracy"] = k / len(dec)
        out["directional_n"] = len(dec)
        lo, hi = wilson_interval(k, len(dec))
        out["accuracy_ci95"] = (round(lo, 3), round(hi, 3))
        for lab in ("UP", "DOWN"):
            pred = [r for r in dec if r["argmax"] == lab]
            actual = [r for r in rows if r["outcome_label"] == lab]
            tp = sum(1 for r in pred if r["outcome_label"] == lab)
            out[f"precision_{lab}"] = tp / len(pred) if pred else None
            out[f"recall_{lab}"] = tp / len(actual) if actual else None
        # trade-usefulness proxy (gross vs net of assumed cost); overlapping/correlated forecasts -> treat with caution
        gross = [(1 if r["argmax"] == "UP" else -1) * r["ret_actual"] for r in dec]
        net = [g - cost_bps_map.get(r["asset"], 2.0) / 1e4 for g, r in zip(gross, dec)]
        out["mean_gross_return"], out["mean_net_return"] = float(np.mean(gross)), float(np.mean(net))
    pr = [r for r in rows if r["probs"]]
    if pr:
        P = np.array([[r["probs"][l] for l in LABELS] for r in pr]); Y = np.array([[1.0 if r["outcome_label"] == l else 0.0 for l in LABELS] for r in pr])
        out["brier"] = float(((P - Y) ** 2).sum(1).mean())
        bp = [r for r in pr if r["baseline"]]
        if bp:
            B = np.array([[r["baseline"][l] for l in LABELS] for r in bp]); Yb = np.array([[1.0 if r["outcome_label"] == l else 0.0 for l in LABELS] for r in bp])
            out["baseline_brier"] = float(((B - Yb) ** 2).sum(1).mean())
            Pb = np.array([[r["probs"][l] for l in LABELS] for r in bp])
            out["brier_matched"] = float(((Pb - Yb) ** 2).sum(1).mean())
            out["brier_skill"] = 1 - out["brier_matched"] / out["baseline_brier"] if out["baseline_brier"] else None
        # calibration of the top-class probability
        top = np.array([P[i].max() for i in range(len(pr))]); hit = np.array([LABELS[int(P[i].argmax())] == pr[i]["outcome_label"] for i in range(len(pr))])
        bins = [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]
        cal = []
        for lo, hi in bins:
            m = (top >= lo) & (top < hi)
            if m.any():
                cal.append({"bin": f"{lo:.1f}-{min(hi,1):.1f}", "n": int(m.sum()), "mean_pred": float(top[m].mean()), "observed": float(hit[m].mean())})
        out["calibration_top_class"] = cal
    return out


def compute_metrics(store, cfg) -> Dict:
    rows = []
    for r in store.evaluated_predictions():
        d = dict(r)
        d["probs"] = json.loads(r["probs"] or "{}"); d["baseline"] = json.loads(r["baseline"] or "{}")
        rows.append(d)
    costs = {a: m.get("cost_bps", 2.0) for a, m in cfg.all_assets().items()}
    res = {"overall": _group_metrics(rows, costs), "pending": len(store.pending_predictions()),
           "note": "Forecasts for different assets/horizons around the same event are correlated; effective independent sample is closer to n_events."}
    for key in ("asset", "category", "horizon_s", "status_label", "stage"):
        g = defaultdict(list)
        for r in rows:
            g[r[key]].append(r)
        res[f"by_{key}"] = {str(k): _group_metrics(v, costs) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}
    hv = [r for r in rows if r["high_vol"]]; lv = [r for r in rows if not r["high_vol"]]
    res["by_volatility"] = {"high_vol": _group_metrics(hv, costs), "normal_vol": _group_metrics(lv, costs)}
    return res


def deteriorated(store, cfg) -> bool:
    """True if recent *validated* live predictions are worse than baseline (enough samples required)."""
    rc = cfg.raw["risk"]
    rows = [dict(r) for r in store.evaluated_predictions() if r["status_label"] == "VALIDATED"]
    rows = sorted(rows, key=lambda r: r["evaluated_ts"] or 0)[-int(rc["deterioration_min_n"]) * 2:]
    if len(rows) < int(rc["deterioration_min_n"]):
        return False
    for r in rows:
        r["probs"] = json.loads(r["probs"] or "{}"); r["baseline"] = json.loads(r["baseline"] or "{}")
    m = _group_metrics([r for r in rows if r["probs"] and r["baseline"]], {})
    if "brier_matched" not in m:
        return False
    return m["brier_matched"] > m["baseline_brier"] + float(rc["deterioration_margin"])


def format_summary(m: Dict) -> str:
    o = m["overall"]
    if o["n"] == 0:
        return "ACCURACY SUMMARY\nNo evaluated predictions yet (pending: %d). Nothing to report - no accuracy is claimed." % m["pending"]
    lines = ["ACCURACY SUMMARY (live, out-of-sample)", f"Evaluated: {o['n']} forecasts from {o['n_events']} events; pending: {m['pending']}"]
    if "directional_accuracy" in o:
        lines.append(f"Directional calls: {o['directional_n']} (coverage {o['coverage_directional']:.0%}); accuracy {o['directional_accuracy']:.0%} "
                     f"(95% CI {o['accuracy_ci95'][0]:.0%}-{o['accuracy_ci95'][1]:.0%})")
        lines.append(f"Mean return per call: gross {o['mean_gross_return']*1e4:+.1f} bps, net of assumed cost {o['mean_net_return']*1e4:+.1f} bps")
    else:
        lines.append("No directional (UP/DOWN) calls made - system abstained everywhere.")
    if "brier_skill" in o and o["brier_skill"] is not None:
        lines.append(f"Brier {o['brier_matched']:.3f} vs baseline {o['baseline_brier']:.3f} (skill {o['brier_skill']:+.1%})")
    if o["n_events"] < 30:
        lines.append("Sample is small: treat all figures as anecdotal.")
    lines.append("Not financial advice. Past performance does not guarantee future results.")
    return "\n".join(lines)
