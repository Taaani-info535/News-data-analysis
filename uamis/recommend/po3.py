"""Read-only alignment check between the user's separate PO3 signal and the latest UAMIS fundamental preference.
No broker connection, no orders, no strategy changes: this only compares two labels. A conflict never 'invalidates' the PO3 signal."""
from __future__ import annotations
import json
from typing import Dict

from .models import AWAY, HZ_NAME, LONG, SHORT, WAIT

ALIGNED, CONFLICTING, NO_PREF, AWAY_R, UNAVAILABLE = ("DIRECTIONALLY ALIGNED", "CONFLICTING", "NO CLEAR FUNDAMENTAL PREFERENCE",
                                                      "STAY AWAY (fundamental / event risk)", "UNAVAILABLE")


def alignment(po3_side: str, direction: str) -> str:
    side = po3_side.strip().upper()
    if side not in ("LONG", "SHORT"):
        raise ValueError("PO3 side must be LONG or SHORT")
    if direction == AWAY:
        return AWAY_R
    if direction == WAIT:
        return NO_PREF
    if (side == "LONG" and direction == LONG) or (side == "SHORT" and direction == SHORT):
        return ALIGNED
    return CONFLICTING


def check(store, instrument: str, po3_side: str, now_ts: int, max_age_minutes: int = 180, horizon_s: int = 3600) -> Dict:
    raw = store.kv_get("rec_latest")
    inst = instrument.upper().replace("/", "")
    if not raw:
        return {"instrument": inst, "po3": po3_side.upper(), "alignment": UNAVAILABLE, "note": "no UAMIS recommendation has been generated yet"}
    data = json.loads(raw)
    age_min = (now_ts - data["generated_at"]) / 60
    recs = data["recs"].get(inst)
    if recs is None:
        return {"instrument": inst, "po3": po3_side.upper(), "alignment": UNAVAILABLE, "note": "instrument not covered by UAMIS"}
    if age_min > max_age_minutes:
        return {"instrument": inst, "po3": po3_side.upper(), "alignment": UNAVAILABLE, "note": f"latest recommendation is {int(age_min)} min old (> {max_age_minutes})"}
    r = recs[str(horizon_s)]
    return {"instrument": inst, "po3": po3_side.upper(), "uamis_1h": r["direction"], "score": r["score"], "confidence": r["confidence"],
            "status": r["status"], "alignment": alignment(po3_side, r["direction"]), "age_minutes": round(age_min),
            "other_horizons": {HZ_NAME[int(h)]: v["direction"] for h, v in recs.items() if int(h) != horizon_s},
            "note": "Context only: UAMIS never invalidates or generates PO3 trades; a conflict simply means no fundamental support for that side."}
