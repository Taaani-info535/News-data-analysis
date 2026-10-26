"""Historical event study: find comparable past events and report observed outcome frequencies with sample size and uncertainty.
Only samples whose outcome window completed BEFORE the target's feature time are eligible (no look-ahead)."""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np

from .features import Sample
from .models import LABELS
from .util import wilson_interval

SIM_COLS_PRE = ["z24", "vol_pct", "fcst_vs_prev"]
SIM_COLS_POST = ["z24", "vol_pct", "surprise_rel", "early_z"]


def study(target_feats: Dict[str, float], target_category: str, asset: str, stage: str, horizon_s: int,
          target_feat_ts: int, hist: List[Sample], min_samples: int = 20, k: int = 40, z: float = 1.96,
          exclude_event_id: Optional[str] = None) -> Dict:
    cols = SIM_COLS_POST if stage == "post" else SIM_COLS_PRE
    pool = [s for s in hist if s.asset == asset and s.stage == stage and s.horizon_s == horizon_s and s.label is not None
            and s.end_ts <= target_feat_ts and s.event_id != exclude_event_id]
    same_cat = [s for s in pool if s.category == target_category]
    out = {"n_pool": len(pool), "n_same_category": len(same_cat), "n": 0, "counts": {l: 0 for l in LABELS}, "freq": {}, "ci": {},
           "status": "INSUFFICIENT HISTORICAL DATA", "note": ""}
    if len(same_cat) < min_samples:
        out["note"] = f"only {len(same_cat)} comparable {target_category} events for {asset} (need {min_samples})"
        return out
    X = np.array([[s.features.get(c, 0.0) for c in cols] for s in same_cat], dtype=float)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    t = np.array([target_feats.get(c, 0.0) for c in cols], dtype=float)
    d = np.sqrt((((X - t) / sd) ** 2).sum(axis=1))
    order = np.argsort(d, kind="stable")[:min(k, len(same_cat))]
    chosen = [same_cat[i] for i in order]
    n = len(chosen)
    counts = {l: sum(1 for s in chosen if s.label == l) for l in LABELS}
    out.update(n=n, counts=counts, status="OK",
               freq={l: counts[l] / n for l in LABELS}, ci={l: wilson_interval(counts[l], n, z) for l in LABELS},
               note=f"{n} nearest of {len(same_cat)} same-category events (similarity on {', '.join(cols)}); overlapping/clustered events not independent")
    return out


def scenario_stats(hist_post: List[Sample], asset: str, category: str, horizon_s: int, sign: int, before_ts: int, min_samples: int = 20,
                   exclude_event_id: Optional[str] = None) -> Dict:
    """Post-stage history conditioned on surprise direction, used for pre-news scenario text."""
    rows = [s for s in hist_post if s.asset == asset and s.category == category and s.horizon_s == horizon_s and s.label is not None
            and s.end_ts <= before_ts and int(s.features.get("surprise_sign", 0)) == sign
            and s.event_id != exclude_event_id]
    n = len(rows)
    if n < min_samples:
        return {"n": n, "status": "INSUFFICIENT HISTORICAL DATA"}
    counts = {l: sum(1 for s in rows if s.label == l) for l in LABELS}
    return {"n": n, "status": "OK", "freq": {l: counts[l] / n for l in LABELS}, "counts": counts,
            "ci": {l: wilson_interval(counts[l], n) for l in LABELS}}
