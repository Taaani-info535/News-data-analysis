"""One recommendation cycle: decide what is due, refresh needed data, run the engine, send ONE consolidated message, record state.
Idempotent: stages are marked processed only after a successful send; identical content is never resent; a scan report is sent only when
the 1h/4h direction classes or currency outlooks changed. Dry-run prints but records nothing."""
from __future__ import annotations
import hashlib
import json
import logging
import os
from typing import Dict, List

from .. import alerts, evaluate, features, pipeline
from ..symbols import asset_kind
from . import report, schedule
from .crypto import CryptoContext, fetch_global
from .engine import Engine
from .models import LONG, SHORT

log = logging.getLogger("uamis.recommend")


def _focus_assets(cfg, due: List[schedule.DueItem]) -> List[str]:
    """Only what the due events need: affected instruments + DXY + index futures (risk sentiment) + crypto (market context)."""
    assets = cfg.all_assets()
    need = set()
    for d in due:
        need.update(features.relevant_assets_cfg(cfg, d.event.currency))
    need.update(n for n, m in assets.items() if asset_kind(m) in ("index", "dollar_index", "crypto"))
    return sorted(need & set(assets))


def _export(cfg, store, result: dict, now_ts: int):
    compact = {"generated_at": now_ts, "recs": {n: {str(h): {"direction": r.direction, "score": r.score, "confidence": r.confidence,
                                                              "status": r.status, "stage": r.stage, "event": r.event, "freshness": r.freshness}
                                                   for h, r in by.items()} for n, by in result["recs"].items()}}
    store.kv_set("rec_latest", json.dumps(compact))
    if cfg.raw["recommend"].get("export_json", True):
        d = os.path.join(os.path.dirname(cfg.db_path) or ".", "reports")
        os.makedirs(d, exist_ok=True)
        full = {"generated_at": now_ts, "note": "Directional filter only. Not a trade signal. UAMIS never trades.",
                "validation": result["validation"], "currency": {c: o.__dict__ for c, o in result["currency"].items()},
                "recs": {n: {str(h): r.to_dict() for h, r in by.items()} for n, by in result["recs"].items()}}
        with open(os.path.join(d, "recommendations_latest.json"), "w") as f:
            json.dump(full, f, indent=2, default=str)


def _store_directional(store, result: dict, now_ts: int):
    n = 0
    for name, by in result["recs"].items():
        for h, r in by.items():
            if r.direction in (LONG, SHORT):
                rid = hashlib.sha1(f"{name}|{h}|{now_ts}".encode()).hexdigest()[:16]
                n += int(store.insert_recommendation({"rec_id": rid, "created_ts": now_ts, "instrument": name, "horizon_s": h, "direction": r.direction,
                                                      "score": r.score, "confidence": r.confidence, "status": r.status, "stage": r.stage,
                                                      "event_name": r.event, "thr": r.thr, "vol_pct": r.vol_pct, "grp": r.group, "components": r.components}))
    return n


def compute(cfg, store, http, now_ts: int, due: List[schedule.DueItem], mds=None, want_hist: bool = True) -> dict:
    mds = mds or pipeline.load_market(cfg, store)
    hist_post = None
    if want_hist:
        rc = cfg.raw["recommend"]
        recent = [e for e in store.events_between(now_ts - int(rc["lookback_hours"]) * 3600, now_ts + 1, int(rc["min_event_importance"]))
                  if e.actual is not None and e.forecast is not None]
        if recent:
            hist_post = pipeline.build_history(cfg, store, mds, now_ts, stages=("post",))
    cx = CryptoContext(cfg, store, mds, now_ts, fetch_global(cfg, store, http, now_ts))
    return Engine(cfg, store, now_ts, mds, hist_post, evaluate.deteriorated(store, cfg), cx).run()


def run_cycle(cfg, store, http, sender, now_ts: int, dry_run: bool = False, out=print, skip_collect: bool = False) -> Dict:
    rc = cfg.raw["recommend"]
    if not rc.get("enabled", True):
        return {"enabled": False}
    res: Dict = {"enabled": True, "sent": False, "mode": "none", "stored": 0}
    t_from = now_ts - int(rc["post_window_minutes"]) * 60 - 60
    t_to = now_ts + max(int(m) for m in rc["pre_checks_minutes"]) * 60 + 1
    due: List[schedule.DueItem] = []
    for ev in store.events_between(t_from, t_to, int(rc["alert_min_importance"])):
        if not features.relevant_assets_cfg(cfg, ev.currency):
            continue
        d = schedule.due_item(store, ev, now_ts, rc)
        if d:
            due.append(d)
    if due and http is not None and not skip_collect:
        pipeline.collect_prices(cfg, store, http, now_ts, assets=_focus_assets(cfg, due), resolutions=[300, 3600])
    scan_on = bool(rc["scan"].get("enabled", True))
    if not due and not scan_on:
        return res
    result = compute(cfg, store, http, now_ts, due)
    _export(cfg, store, result, now_ts)
    mat = report.material(result, due)
    if due:
        text = report.build_report(cfg, result, due, now_ts)
        key = hashlib.sha1("|".join(sorted(d.event.event_id + d.key for d in due)).encode()).hexdigest()[:12]
        r = alerts.dispatch(store, cfg, sender, f"rec:{key}", "rec_event", text, mat, now_ts, dry_run, out)
        res.update(mode="event", result=r, stages=[d.key for d in due])
        if r == "sent":
            for d in due:
                schedule.mark_processed(store, d, now_ts)
            res["stored"] = _store_directional(store, result, now_ts)
            res["sent"] = True
            # the report just sent is now the baseline for "meaningful change": a scan must not re-send the same picture minutes later
            base = alerts.digest(report.material(result, []))
            store.alert_record(f"rec:scan:{base}", "rec:scan", "scan", now_ts, base)
    else:
        prev = store.alert_last("rec:scan", "scan")
        changes = []
        if prev is not None:
            changes = ["directional classes changed vs previous scan report"]
        text = report.build_report(cfg, result, [], now_ts, scan_changes=changes)
        interval = int(rc["scan"]["min_interval_minutes"]) * 60
        # first-ever scan is sent only if it contains at least one directional preference (avoid an all-WAIT report on day one)
        has_dir = any(r.direction in (LONG, SHORT) for by in result["recs"].values() for r in by.values())
        if prev is None and not has_dir:
            res.update(mode="scan", result="skipped_no_preference")
        else:
            r = alerts.dispatch(store, cfg, sender, "rec:scan", "scan", text, mat, now_ts, dry_run, out, cooldown_s=interval)
            res.update(mode="scan", result=r)
            if r == "sent":
                res["stored"] = _store_directional(store, result, now_ts); res["sent"] = True
    res["n_directional"] = sum(1 for by in result["recs"].values() for r in by.values() if r.direction in (LONG, SHORT))
    return res
