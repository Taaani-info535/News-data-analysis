"""Orchestration of all jobs. Every job is idempotent: re-running after a crash or duplicate workflow never double-sends or double-logs."""
from __future__ import annotations
import csv
import json
import logging
import os
import re
from typing import Dict, List, Optional

from . import alerts, evaluate, features, ml, predict
from .adapters import calendar_ff, fred, llm, news_rss, prices_binance, prices_yahoo
from .adapters.calendar_ff import make_event_id
from .classify import classify_event, hawkish_rule, surprise_sign, theoretical_impact
from .config import Config
from .models import Event
from .prices import MarketData
from .storage import Store
from .symbols import validate_all
from .util import parse_iso_to_ts, parse_value, ts_to_iso

log = logging.getLogger("uamis.pipeline")


def _range_seconds(r: str) -> int:
    m = re.fullmatch(r"(\d+)([dy])", r)
    n, u = int(m.group(1)), m.group(2)
    return n * (86400 if u == "d" else 365 * 86400)


# ----------------------------------------------------------------------------- collection
def collect_prices(cfg: Config, store: Store, http, now_ts: int, backfill: bool = False, assets: Optional[List[str]] = None,
                   resolutions: Optional[List[int]] = None) -> Dict:
    from .validation import validate_candles
    ccfg = cfg.raw["candles"]
    ranges = ccfg["yahoo_backfill_range"] if backfill else ccfg["yahoo_range"]
    summary = {"assets_ok": 0, "assets_failed": {}, "rows": 0, "rejected": 0}
    all_assets = cfg.all_assets()
    invalid = validate_all(all_assets)
    for a in (assets or sorted(all_assets)):
        meta = all_assets.get(a)
        if not meta:
            continue
        if a in invalid:                       # malformed symbol: reported, never sent to the provider
            summary["assets_failed"][a] = "invalid symbol: " + "; ".join(invalid[a])[:100]
            continue
        ok = True
        for res in ccfg["resolutions"]:
            res = int(res)
            if resolutions is not None and res not in resolutions:
                continue
            try:
                if meta["source"] == "yahoo":
                    raw = prices_yahoo.fetch_candles(http, meta["symbol"], res, ranges[res], now_ts)
                elif meta["source"] == "binance":
                    raw = prices_binance.fetch_candles(http, meta["symbol"], res, _range_seconds(ranges[res]), now_ts,
                                                       base=cfg.g("binance_base_url", default=prices_binance.BASE))
                else:
                    raise ValueError(f"unknown source {meta['source']}")
                clean, rej = validate_candles(raw, res, now_ts)
                summary["rows"] += store.upsert_candles(a, res, clean)
                summary["rejected"] += rej
            except Exception as e:           # a failing symbol must not stop others; failure is reported, not hidden
                ok = False
                summary["assets_failed"][f"{a}@{res}"] = f"{type(e).__name__}: {str(e)[:80]}"
        summary["assets_ok"] += int(ok)
    if not assets:
        for res, days in ccfg["retain_days"].items():
            store.prune_candles(int(res), now_ts - int(days) * 86400)
        if os.environ.get("FRED_API_KEY"):
            for name, sid in fred.SERIES.items():
                try:
                    store.upsert_candles(name, 86400, fred.fetch_series(http, sid, ts_to_iso(now_ts - 400 * 86400)[:10]))
                except Exception as e:
                    summary["assets_failed"][name] = type(e).__name__
    log.info("collect: %s", summary)
    return summary


def refresh_calendar(cfg: Config, store: Store, http, now_ts: int) -> Dict:
    try:
        evs = calendar_ff.fetch_calendar(http, cfg.raw["calendar"]["feed_urls"], now_ts)
    except Exception as e:
        log.warning("calendar refresh failed: %s", type(e).__name__)
        return {"error": type(e).__name__}
    stats = {"new": 0, "updated": 0, "same": 0}
    for e in evs:
        stats[store.upsert_event(e)] += 1
    return stats


def refresh_news(cfg: Config, store: Store, http, sender, now_ts: int, dry_run: bool, out=print) -> Dict:
    n = cfg.raw["news"]
    items = news_rss.fetch_news(http, n["feeds"], now_ts, n["max_age_hours"])
    stats = {"new": 0, "alerted": 0}
    assets = cfg.all_assets()
    for it in items:
        if not store.insert_news(it):
            continue
        stats["new"] += 1
        if it.category in n["alert_categories"]:
            aff = sorted(a for a, m in assets.items() if "USD" in (m.get("base"), m.get("quote")) or m.get("base") in ("IDX", "XAU", "XAG", "BTC", "ETH"))
            r = alerts.dispatch(store, cfg, sender, f"news:{it.news_id}", "news", alerts.format_news(it, aff), [it.news_id], now_ts, dry_run, out)
            stats["alerted"] += int(r in ("sent", "dry_run"))
    return stats


# ----------------------------------------------------------------------------- history / training
def load_market(cfg: Config, store: Store) -> Dict[str, MarketData]:
    return {a: MarketData.from_store(store, a) for a in cfg.all_assets()}


def build_history(cfg: Config, store: Store, mds, now_ts: int, stages=("pre", "post")):
    evs = store.all_events_before(now_ts)
    return {(st, h * 60): features.build_samples(evs, mds, st, h * 60, cfg, now_ts) for st in stages for h in cfg.horizons}


def model_version(now_ts: int, n: int) -> str:
    return f"{ts_to_iso(now_ts)[:10]}-n{n}"


def train_all(cfg: Config, store: Store, now_ts: int) -> Dict:
    mds = load_market(cfg, store)
    hist = build_history(cfg, store, mds, now_ts)
    names = sorted(cfg.all_assets())
    reports = {}
    for (st, h), smp in hist.items():
        reports[f"{st}_{h}"] = ml.train(smp, st, h, names, cfg, model_version(now_ts, len(smp)), cfg.model_dir)
    os.makedirs(cfg.model_dir, exist_ok=True)
    with open(os.path.join(cfg.model_dir, "training_summary.json"), "w") as f:
        json.dump({"trained_at": ts_to_iso(now_ts), "reports": reports}, f, indent=2, default=str)
    return reports


# ----------------------------------------------------------------------------- tick
def _pred_rows(store: Store, event_id: str, stage: str):
    return store.db.execute("SELECT * FROM predictions WHERE event_id=? AND stage=?", (event_id, stage)).fetchall()


def _group(preds) -> Dict[str, Dict[int, "predict.Prediction"]]:
    d: Dict[str, Dict[int, object]] = {}
    for p in preds:
        d.setdefault(p.asset, {})[p.horizon_s] = p
    return d


def _other_event_minutes(store: Store, ev: Event) -> Optional[float]:
    near = [e for e in store.events_between(ev.t0 - 5400, ev.t0 + 5400, 3) if e.event_id != ev.event_id and abs(e.t0 - ev.t0) > 60]
    return min((abs(e.t0 - ev.t0) / 60 for e in near), default=None)


def market_context(store: Store, cfg: Config, ev: Event, now_ts: int) -> List[str]:
    lines = []
    if "DXY" in cfg.all_assets():
        md = MarketData.from_store(store, "DXY")
        a, b = md.price_at(ev.t0, 300), md.price_at(now_ts, 300)
        if a and b:
            lines.append(f"Context: DXY since release {(b[0] / a[0] - 1) * 100:+.2f}%")
    y = store.load_candles("US10Y", 86400)
    if len(y) >= 2:
        lines.append(f"Context: US10Y last daily obs {y[-1]['close']:.2f}% (prev {y[-2]['close']:.2f}%; daily data, not intraday)")
    return lines


def _recommend(cfg: Config, store: Store, http, sender, now_ts: int, dry_run: bool, out, skip_collect: bool) -> Dict:
    """Recommendation engine cycle. Isolated: a failure here must never break calendar/news/prediction alerts."""
    try:
        from .recommend.cycle import run_cycle
        return run_cycle(cfg, store, http, sender, now_ts, dry_run, out, skip_collect)
    except Exception as e:                                   # logged, reported in the tick result, never swallowed silently
        log.exception("recommendation cycle failed")
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}


def run_tick(cfg: Config, store: Store, http, sender, now_ts: int, dry_run: bool = False, out=print, skip_collect: bool = False) -> Dict:
    res: Dict = {"calendar": refresh_calendar(cfg, store, http, now_ts) if http is not None else {"skipped": True}}
    if http is not None:
        res["news"] = refresh_news(cfg, store, http, sender, now_ts, dry_run, out)
    c = cfg.raw["calendar"]
    assets = cfg.all_assets()
    pairs, fx_msg = cfg.forex_status()
    res.update(heads_up=0, pre=0, post=0, skipped_dupes=0)
    # 1) heads-up
    for ev in store.events_between(now_ts, now_ts + int(c["heads_up_lead_minutes"]) * 60, int(c["alert_min_importance"])):
        rel = features.relevant_assets_cfg(cfg, ev.currency)
        if not rel:
            continue
        r = alerts.dispatch(store, cfg, sender, ev.event_id, "heads_up", alerts.format_upcoming(ev, rel, now_ts), [ev.event_id], now_ts, dry_run, out)
        res["heads_up"] += int(r in ("sent", "dry_run")); res["skipped_dupes"] += int(r in ("duplicate", "cooldown"))
    pre_evs = store.events_between(now_ts, now_ts + int(c["pre_lead_minutes"]) * 60, int(c["alert_min_importance"]))
    post_evs = [e for e in store.events_between(now_ts - int(c["post_max_lag_minutes"]) * 60, now_ts, int(c["alert_min_importance"]))
                if e.actual is not None]
    legacy = bool(cfg.g("recommend", "legacy_event_reports", default=False))   # older per-event probability messages (ledger logging is unaffected)
    if not pre_evs and not post_evs:
        res["recommend"] = _recommend(cfg, store, http, sender, now_ts, dry_run, out, skip_collect)
        return res
    mds = load_market(cfg, store)
    hist = None
    det = evaluate.deteriorated(store, cfg)
    horizons = [h * 60 for h in cfg.horizons]

    def get_hist():
        nonlocal hist
        if hist is None:
            hist = build_history(cfg, store, mds, now_ts)
        return hist

    # 2) pre-event
    for ev in pre_evs:
        rel = features.relevant_assets_cfg(cfg, ev.currency)
        if not rel:
            continue
        rows = _pred_rows(store, ev.event_id, "pre")
        retry_missing = bool(rows) and any(r["status_label"] == "INSUFFICIENT DATA" and r["eval_status"] == "PENDING" for r in rows)
        if not rows or retry_missing:
            H = get_hist(); om = _other_event_minutes(store, ev)
            for a in rel:
                for h in horizons:
                    p = predict.predict_one(cfg, None, ev, a, "pre", h, mds[a], now_ts, H[("pre", h)], det, om)
                    store.insert_prediction(p)
                    store.retry_insufficient_prediction(p)
            rows = _pred_rows(store, ev.event_id, "pre")
        preds = [predict.prediction_from_row(r) for r in rows]
        if not legacy:
            continue                                   # ledger rows are logged above; the consolidated report replaces this message
        grouped = _group(preds)
        focus = [a for a in rel if a not in pairs]
        H = get_hist()
        scen = {a: predict.scenario_lines(cfg, ev, a, H[("post", 3600)], 3600, now_ts) for a in focus}
        text = alerts.format_pre(cfg, ev, grouped, scen, focus, pairs, fx_msg, rel, now_ts)
        mat = [f"{p.asset}:{p.horizon_s}:{p.argmax}:{p.decision}:{p.status_label}" for p in sorted(preds, key=lambda x: (x.asset, x.horizon_s))]
        r = alerts.dispatch(store, cfg, sender, ev.event_id, "pre", text, mat, now_ts, dry_run, out)
        res["pre"] += int(r in ("sent", "dry_run")); res["skipped_dupes"] += int(r in ("duplicate", "cooldown"))
    # 3) post-event
    if post_evs and not skip_collect and http is not None:
        need = sorted({a for e in post_evs for a in features.relevant_assets_cfg(cfg, e.currency)})
        collect_prices(cfg, store, http, now_ts, assets=need)
        mds = load_market(cfg, store)
    for ev in post_evs:
        rel = features.relevant_assets_cfg(cfg, ev.currency)
        if not rel:
            continue
        seen = ev.actual_seen_at or now_ts
        anchor = max((seen // 300) * 300, ev.t0 + int(c["post_delay_minutes"]) * 60)
        if anchor > now_ts:
            continue
        rows = _pred_rows(store, ev.event_id, "post")
        retry_missing = bool(rows) and any(r["status_label"] == "INSUFFICIENT DATA" and r["eval_status"] == "PENDING" for r in rows)
        if not rows or retry_missing:
            H = get_hist(); om = _other_event_minutes(store, ev)
            for a in rel:
                for h in horizons:
                    if now_ts >= anchor + h:
                        continue          # window already over: cannot forecast the past
                    p = predict.predict_one(cfg, None, ev, a, "post", h, mds[a], now_ts, H[("post", h)], det, om, feat_ts=anchor, anchor_ts=anchor)
                    store.insert_prediction(p)
                    store.retry_insufficient_prediction(p)
            rows = _pred_rows(store, ev.event_id, "post")
        preds = [predict.prediction_from_row(r) for r in rows]
        grouped = _group(preds)
        focus = [a for a in rel if a not in pairs]
        sg = surprise_sign(ev.actual, ev.forecast); hk = hawkish_rule(ev.category, ev.name)
        theory = {a: theoretical_impact(ev.currency, hk, sg, assets[a]) for a in focus}
        if not legacy:
            continue
        note = "" if preds else "No forecast logged: the outcome window had already elapsed or market data was unavailable."
        text = alerts.format_post(cfg, ev, grouped, focus, pairs, fx_msg, rel, theory, market_context(store, cfg, ev, now_ts), now_ts, note)
        summ = llm.summarize(http, cfg, text) if http is not None else None
        if summ:
            text = summ + "\n\n" + text
        mat = [ev.event_id, ev.actual_raw] + [f"{p.asset}:{p.horizon_s}:{p.argmax}:{p.decision}" for p in sorted(preds, key=lambda x: (x.asset, x.horizon_s))]
        r = alerts.dispatch(store, cfg, sender, ev.event_id, "post", text, mat, now_ts, dry_run, out)
        res["post"] += int(r in ("sent", "dry_run")); res["skipped_dupes"] += int(r in ("duplicate", "cooldown"))
    res["recommend"] = _recommend(cfg, store, http, sender, now_ts, dry_run, out, skip_collect)
    return res


def run_evaluate(cfg: Config, store: Store, sender, now_ts: int, send_summary: bool, dry_run: bool, out=print) -> Dict:
    from .recommend import validation as recval
    sc = evaluate.score_pending(store, cfg, now_ts)
    m = evaluate.compute_metrics(store, cfg)
    rsc = recval.score_pending(store, cfg, now_ts)
    rm = recval.metrics(store, cfg)
    os.makedirs(os.path.dirname(cfg.db_path) or ".", exist_ok=True)
    rp = os.path.join(os.path.dirname(cfg.db_path) or ".", "reports")
    os.makedirs(rp, exist_ok=True)
    with open(os.path.join(rp, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2, default=str)
    with open(os.path.join(rp, "recommendation_metrics.json"), "w") as f:
        json.dump(rm, f, indent=2, default=str)
    txt = evaluate.format_summary(m)
    rtxt = recval.format_summary(rm)
    if send_summary:
        day = ts_to_iso(now_ts)[:10]
        alerts.dispatch(store, cfg, sender, f"summary:{day}", "summary", txt + "\n\n" + rtxt, [day, str(m["overall"]["n"]), str(rm["overall"]["n"])], now_ts, dry_run, out)
    return {"scored": sc, "summary": txt, "recommendations_scored": rsc, "recommendation_summary": rtxt}


# ----------------------------------------------------------------------------- manual data helpers
def parse_importance(v: str) -> int:
    v = str(v).strip().lower()
    return {"low": 1, "medium": 2, "med": 2, "high": 3}.get(v) or int(v)


def import_events_csv(store: Store, path: str, now_ts: int) -> Dict:
    """CSV columns: name,currency,datetime_utc,importance,forecast,previous,actual  (blank = unknown; never guessed)."""
    n_ok = n_bad = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t0 = parse_iso_to_ts(row["datetime_utc"]); name = row["name"].strip(); cur = row["currency"].strip().upper()
                cat, _ = classify_event(name)
                store.upsert_event(Event(make_event_id(cur, name, t0), name, cur, t0, parse_importance(row["importance"]), cat,
                                         row.get("forecast", "") or "", row.get("previous", "") or "", row.get("actual", "") or "",
                                         parse_value(row.get("forecast")), parse_value(row.get("previous")), parse_value(row.get("actual")),
                                         "csv-import", now_ts, now_ts, t0 if parse_value(row.get("actual")) is not None else None))
                n_ok += 1
            except (KeyError, ValueError, TypeError):
                n_bad += 1
    return {"imported": n_ok, "rejected": n_bad}


def set_actual(store: Store, event_id: str, actual_raw: str, now_ts: int) -> bool:
    ev = store.get_event(event_id)
    v = parse_value(actual_raw)
    if ev is None or v is None:
        return False
    ev.actual, ev.actual_raw, ev.actual_seen_at, ev.updated_at = v, actual_raw, now_ts, now_ts
    store.upsert_event(ev)
    return True
