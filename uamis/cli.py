"""Command line: python -m uamis <command>"""
from __future__ import annotations
import argparse
import json
import logging
import sys

from .adapters.http import HttpClient
from .alerts import TelegramSender
from .config import load_config
from .storage import Store
from .util import utcnow_ts, ts_to_iso
from . import pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="uamis", description="Universal AI Market Intelligence System (intelligence only; never trades)")
    ap.add_argument("--config", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("status", help="show data counts, forex gate and model status")
    c = sp.add_parser("collect", help="download/store candles"); c.add_argument("--backfill", action="store_true")
    t = sp.add_parser("tick", help="calendar+news refresh, heads-up/pre/post alerts"); t.add_argument("--dry-run", action="store_true")
    sp.add_parser("train", help="train/validate models from stored history")
    e = sp.add_parser("evaluate", help="score finished predictions"); e.add_argument("--send-summary", action="store_true"); e.add_argument("--dry-run", action="store_true")
    i = sp.add_parser("import-events", help="import historical events from CSV"); i.add_argument("csv")
    a = sp.add_parser("set-actual", help="manually set an event's actual value"); a.add_argument("event_id"); a.add_argument("actual")
    sp.add_parser("demo", help="offline end-to-end run on SYNTHETIC data (meaningless numbers; installation check only)")
    r = sp.add_parser("recommend", help="print the consolidated market-direction report now (never trades)"); r.add_argument("--send", action="store_true", help="also send it to Telegram"); r.add_argument("--no-collect", action="store_true")
    po = sp.add_parser("po3-check", help="compare your PO3 signal side with the latest UAMIS preference (read-only)"); po.add_argument("instrument"); po.add_argument("side", choices=["LONG", "SHORT", "long", "short"])
    sp.add_parser("events", help="list stored events (id, time, actual) from 1 day ago to 8 days ahead")
    sp.add_parser("telegram-test", help="send a test message to verify bot token/chat id")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    if args.cmd == "demo":
        from .demo import run_demo
        return run_demo()
    cfg = load_config(args.config)
    store = Store(cfg.db_path)
    now = utcnow_ts()
    http = HttpClient.from_config(cfg)
    sender = TelegramSender(cfg.telegram_token, cfg.telegram_chat_id)
    if args.cmd == "status":
        pairs, msg = cfg.forex_status()
        print(msg)
        n = {t: store.db.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"] for t in ("candles", "events", "news", "predictions", "alerts")}
        print("rows:", json.dumps(n)); print("telegram configured:", sender.configured)
        from .symbols import validate_all
        bad = validate_all(cfg.all_assets())
        print("symbol validation:", "all %d symbols valid" % len(cfg.all_assets()) if not bad else bad)
        import os
        for f in sorted(os.listdir(cfg.model_dir)) if os.path.isdir(cfg.model_dir) else []:
            if f.endswith(".json") and f != "training_summary.json":
                r = json.load(open(os.path.join(cfg.model_dir, f)))
                print(f"model {r['stage']}/{r['horizon_s']}s: {r['status']} {r.get('reason','')}")
    elif args.cmd == "collect":
        s = pipeline.collect_prices(cfg, store, http, now, backfill=args.backfill); print(json.dumps(s, indent=2))
    elif args.cmd == "tick":
        print(json.dumps(pipeline.run_tick(cfg, store, http, sender, now, dry_run=args.dry_run), indent=2, default=str))
    elif args.cmd == "train":
        r = pipeline.train_all(cfg, store, now)
        for k, v in r.items():
            print(f"{k}: {v['status']} n={v['n_samples']} {v.get('reason','')}")
    elif args.cmd == "evaluate":
        r = pipeline.run_evaluate(cfg, store, sender, now, args.send_summary, args.dry_run); print(r["summary"]); print(); print(r["recommendation_summary"])
    elif args.cmd == "recommend":
        from .recommend import cycle, report
        if not args.no_collect:
            pipeline.collect_prices(cfg, store, http, now, assets=cycle._focus_assets(cfg, []), resolutions=[300, 3600])
        result = cycle.compute(cfg, store, http, now, [])
        cycle._export(cfg, store, result, now)
        text = report.build_report(cfg, result, [], now)
        print(text)
        if args.send:
            from . import alerts
            print("delivery:", alerts.dispatch(store, cfg, sender, "rec:manual", "manual", text, [str(now)], now, False, print, cooldown_s=0))
    elif args.cmd == "po3-check":
        from .recommend import po3
        print(json.dumps(po3.check(store, args.instrument, args.side, now), indent=2))
    elif args.cmd == "import-events":
        print(pipeline.import_events_csv(store, args.csv, now))
    elif args.cmd == "set-actual":
        print("ok" if pipeline.set_actual(store, args.event_id, args.actual, now) else "failed (unknown event or unparseable value)")
    elif args.cmd == "events":
        for ev in store.events_between(now - 86400, now + 8 * 86400, 1):
            print(f"{ev.event_id}  {ts_to_iso(ev.t0)}  {'*' * ev.importance:<3} {ev.currency} {ev.name}  fcst={ev.forecast_raw or '-'} prev={ev.previous_raw or '-'} actual={ev.actual_raw or '-'}")
    elif args.cmd == "telegram-test":
        if not sender.configured:
            print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"); return 2
        sender.send("UAMIS test message: Telegram delivery works. (No market data in this message.)"); print("sent")
    store.close()
    return 0
