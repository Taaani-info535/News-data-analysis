"""Live validation of the RECOMMENDATIONS themselves (separate from the ML models' own validation).
Directional (LONG/SHORT) preferences are logged when a report is sent and scored after their horizon with the same
volatility-aware sideways threshold used by the prediction ledger. Nothing is called 'validated' without enough scored outcomes."""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..features import label_from_return
from ..prices import MarketData
from ..util import wilson_interval
from .models import HZ_NAME, LONG

NO_DATA_DEADLINE_S = 7 * 86400


def score_pending(store, cfg, now_ts: int) -> Dict[str, int]:
    grace = int(cfg.g("evaluation", "grace_seconds", default=600))
    mds: Dict[str, MarketData] = {}
    res = {"evaluated": 0, "pending": 0, "no_data": 0}
    for r in store.pending_recommendations():
        end = r["created_ts"] + r["horizon_s"]
        if now_ts < end + grace:
            res["pending"] += 1
            continue
        a = r["instrument"]
        mds.setdefault(a, MarketData.from_store(store, a))
        max_res = max(300, r["horizon_s"] // 3)
        pa, pe = mds[a].price_at(r["created_ts"], max_res), mds[a].price_at(end, max_res)
        if r["thr"] is None or not (pa and pe):
            if now_ts > end + NO_DATA_DEADLINE_S:
                store.set_recommendation_outcome(r["rec_id"], None, None, "NO_DATA", now_ts); res["no_data"] += 1
            else:
                res["pending"] += 1
            continue
        ret = math.log(pe[0] / pa[0])
        store.set_recommendation_outcome(r["rec_id"], ret, label_from_return(ret, r["thr"]), "EVALUATED", now_ts)
        res["evaluated"] += 1
    return res


def _stats(rows: List[dict], cost_bps: Dict[str, float]) -> Dict:
    n = len(rows)
    out: Dict = {"n": n}
    if not n:
        return out
    hits = sum(1 for r in rows if (r["direction"] == LONG) == (r["outcome_label"] == "UP") and r["outcome_label"] in ("UP", "DOWN"))
    dec = [r for r in rows if r["outcome_label"] in ("UP", "DOWN")]
    out.update(decisive=len(dec), sideways_share=1 - len(dec) / n, hits=hits)
    if dec:
        out["hit_rate_decisive"] = hits / len(dec)
        lo, hi = wilson_interval(hits, len(dec))
        out["hit_ci95"] = (round(lo, 3), round(hi, 3))
    gross = [(1 if r["direction"] == LONG else -1) * r["ret_actual"] for r in rows]
    net = [g - cost_bps.get(r["instrument"], 2.0) / 1e4 for g, r in zip(gross, rows)]
    out["mean_gross_bps"], out["mean_net_bps"] = float(np.mean(gross)) * 1e4, float(np.mean(net)) * 1e4
    return out


def validation_status(store, cfg, horizon_s: int) -> Dict:
    v = cfg.raw["recommend"]["validation"]
    rows = [dict(r) for r in store.evaluated_recommendations(horizon_s)]
    costs = {a: m.get("cost_bps", 2.0) for a, m in cfg.all_assets().items()}
    st = _stats(rows, costs)
    ok = (st["n"] >= int(v["min_evaluated"]) and st.get("decisive", 0) >= int(v["min_evaluated"]) // 2
          and st.get("hit_ci95", (0, 0))[0] > float(v["min_lower_ci"]) and st.get("mean_net_bps", -1) > 0) if st["n"] else False
    return {"validated": bool(ok), "n": st["n"], "needed": int(v["min_evaluated"]), "stats": st}


def metrics(store, cfg) -> Dict:
    costs = {a: m.get("cost_bps", 2.0) for a, m in cfg.all_assets().items()}
    rows = [dict(r) for r in store.evaluated_recommendations()]
    res = {"overall": _stats(rows, costs), "pending": len(store.pending_recommendations()), "by_horizon": {}, "by_instrument": {}, "by_score_bucket": {},
           "note": "Hit rate is measured on decisive (UP/DOWN) outcomes; SIDEWAYS counts against the preference in n but not in the hit-rate denominator. "
                   "Recommendations around one event are correlated, so effective samples are fewer than n."}
    for key, name in (("horizon_s", "by_horizon"), ("instrument", "by_instrument")):
        g = defaultdict(list)
        for r in rows:
            g[r[key]].append(r)
        res[name] = {(HZ_NAME.get(k, str(k)) if key == "horizon_s" else k): _stats(v, costs) for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}
    b = defaultdict(list)
    for r in rows:
        a = abs(r["score"])
        b["25-40" if a < 40 else "40-60" if a < 60 else "60+"].append(r)
    res["by_score_bucket"] = {k: _stats(v, costs) for k, v in sorted(b.items())}
    res["validation"] = {HZ_NAME[h * 60]: validation_status(store, cfg, h * 60)["validated"] for h in cfg.horizons}
    return res


def format_summary(m: Dict) -> str:
    o = m["overall"]
    if not o["n"]:
        return f"RECOMMENDATION TRACK RECORD\nNo evaluated LONG/SHORT preferences yet (pending: {m['pending']}). No accuracy is claimed."
    L = ["RECOMMENDATION TRACK RECORD (live)", f"Evaluated LONG/SHORT preferences: {o['n']} (pending {m['pending']})"]
    if "hit_rate_decisive" in o:
        L.append(f"Hit rate on decisive outcomes: {o['hit_rate_decisive']:.0%} (95% CI {o['hit_ci95'][0]:.0%}-{o['hit_ci95'][1]:.0%}, n={o['decisive']}); sideways {o['sideways_share']:.0%}")
    L.append(f"Mean return per preference: gross {o['mean_gross_bps']:+.1f} bps, net of assumed cost {o['mean_net_bps']:+.1f} bps")
    L.append("VALIDATED horizons: " + (", ".join(h for h, v in m["validation"].items() if v) or "none yet"))
    if o["n"] < 100:
        L.append("Sample is small: treat as anecdotal.")
    return "\n".join(L)
