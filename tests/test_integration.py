"""Integration tests with MOCKED HTTP responses. These do NOT verify live API behaviour."""
import itertools
import json
import os
import pytest

from uamis import pipeline, alerts
from uamis.adapters.http import HttpError
from uamis.demo import make_demo_config, synth_events, synth_market, DEMO_PAIRS
from uamis.storage import Store
from tests.conftest import END, NOW

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>Fed</title>
<item><title>Federal Reserve issues FOMC statement</title><link>http://fed/x</link><pubDate>{pub}</pubDate></item></channel></rss>"""


def iso(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def rfc(ts):
    from email.utils import formatdate
    return formatdate(ts, usegmt=True)


class FakeHttp:
    def __init__(self, calendar, fail_prices=True, fail_calendar=False):
        self.calendar, self.fail_prices, self.fail_calendar, self.calls = calendar, fail_prices, fail_calendar, []

    def get_json(self, url, **kw):
        self.calls.append(url)
        if "faireconomy" in url:
            if self.fail_calendar:
                raise HttpError("boom")
            return self.calendar
        raise HttpError("price API unavailable (simulated)")

    def get_text(self, url, **kw):
        self.calls.append(url)
        return RSS.format(pub=rfc(NOW - 1200))

    def request(self, *a, **k):
        raise HttpError("n/a")


class Snd:
    configured = True
    def __init__(self): self.sent = []
    def send(self, t, m=3500): self.sent.append(t)


def legacy(cfg):
    """These tests exercise the legacy per-event probability messages, which the consolidated report replaces by default."""
    cfg.raw["recommend"]["legacy_event_reports"] = True
    return cfg


def feed(pre_t0, post_t0):
    return [
        {"title": "CPI y/y", "country": "USD", "date": iso(pre_t0), "impact": "High", "forecast": "3.1%", "previous": "3.0%", "actual": ""},
        {"title": "Non-Farm Employment Change", "country": "USD", "date": iso(post_t0), "impact": "High", "forecast": "180K", "previous": "170K", "actual": "230K"},
    ]


def test_tick_is_idempotent_across_restarts_and_survives_api_failures(small_world, tmp_path):
    cfg, store, _ = small_world
    legacy(cfg)
    snd, out = Snd(), []
    http = FakeHttp(feed(NOW + 1800, NOW - 1200))
    r1 = pipeline.run_tick(cfg, store, http, snd, NOW, out=out.append)
    assert r1["pre"] == 1 and r1["post"] == 1 and r1["heads_up"] >= 1
    n_sent = len(snd.sent)
    npred = store.db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    assert n_sent >= 4 and npred > 0                               # heads-up, pre, post, news
    # second run: same state -> nothing new
    r2 = pipeline.run_tick(cfg, store, http, snd, NOW + 120, out=out.append)
    assert len(snd.sent) == n_sent and r2["pre"] == 0 and r2["post"] == 0
    # "workflow restart": new process = new Store connection on same file
    store2 = Store(cfg.db_path)
    r3 = pipeline.run_tick(cfg, store2, FakeHttp(feed(NOW + 1800, NOW - 1200)), snd, NOW + 240, out=out.append)
    assert len(snd.sent) == n_sent and r3["pre"] == 0
    assert store2.db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"] == npred
    store2.close()


def test_alerts_say_stay_away_when_unvalidated_and_never_claim_probability_validated(small_world):
    cfg, store, _ = small_world
    legacy(cfg)
    snd = Snd()
    pipeline.run_tick(cfg, store, FakeHttp(feed(NOW + 1800, NOW - 1200)), snd, NOW, out=lambda *_: None)
    pre = next(t for t in snd.sent if "Stage: PRE-NEWS" in t)
    assert "STAY AWAY" in pre and "VALIDATED]" not in pre and "result unknown" in pre
    assert "FOREX: FOREX DISABLED" in pre                           # forex list not supplied in this fixture -> explicit, not silent
    post = next(t for t in snd.sent if "Stage: POST-NEWS" in t)
    assert "ABOVE forecast" in post and "textbook impact" in post
    rows = store.db.execute("SELECT DISTINCT status_label, decision FROM predictions").fetchall()
    assert all(r["decision"] != "CONDITIONS ACCEPTABLE (not a trade signal)" for r in rows)   # nothing validated -> no green light


def test_calendar_failure_does_not_crash_tick(small_world):
    cfg, store, _ = small_world
    r = pipeline.run_tick(cfg, store, FakeHttp([], fail_calendar=True), Snd(), NOW, out=lambda *_: None)
    assert "error" in r["calendar"]


def test_incomplete_history_and_missing_prices_produce_honest_report(tmp_path, base_cfg):
    cfg = legacy(make_demo_config(str(tmp_path), pairs=[], base_cfg=base_cfg))
    store = Store(cfg.db_path)                                      # no candles, no events at all
    snd = Snd()
    pipeline.run_tick(cfg, store, FakeHttp(feed(NOW + 1800, NOW - 1200)), snd, NOW, out=lambda *_: None)
    pre = next(t for t in snd.sent if "Stage: PRE-NEWS" in t)
    assert "STAY AWAY" in pre and ("market data" in pre or "insufficient" in pre) and "UP" not in pre.split("Decision")[0].replace("UPCOMING", "")
    store.close()


def test_post_without_actual_vs_forecast_is_not_invented(small_world):
    cfg, store, _ = small_world
    legacy(cfg)
    f = [{"title": "Unemployment Rate", "country": "USD", "date": iso(NOW - 1200), "impact": "High", "forecast": "", "previous": "3.7%", "actual": "3.7%"}]
    snd = Snd()
    pipeline.run_tick(cfg, store, FakeHttp(f), snd, NOW, out=lambda *_: None)
    post = next(t for t in snd.sent if "Stage: POST-NEWS" in t)
    assert "surprise not computable" in post


def test_all_28_pairs_in_every_main_report_when_list_valid(tmp_path, base_cfg):
    cfg = legacy(make_demo_config(str(tmp_path), pairs=DEMO_PAIRS, base_cfg=base_cfg))   # the user's 28 pair names; prices are synthetic
    store = Store(cfg.db_path)
    evs = synth_events(END - 86400, days=40)
    for e in evs: store.upsert_event(e)
    synth_market(cfg, store, END, days=40, events=evs)
    snd = Snd()
    pipeline.run_tick(cfg, store, FakeHttp(feed(NOW + 1800, NOW - 1200)), snd, NOW, out=lambda *_: None)
    reports = [t for t in snd.sent if "Stage: PRE-NEWS" in t or "Stage: POST-NEWS" in t]
    assert len(reports) == 2
    for t in reports:
        assert "FOREX (all 28 configured pairs)" in t
        assert all(p in t for p in DEMO_PAIRS)                      # every one of the 28 appears
        assert "PAIR   | 15M" in t
    store.close()


def test_collect_prices_reports_failures_without_crashing(small_world):
    cfg, store, _ = small_world
    s = pipeline.collect_prices(cfg, store, FakeHttp([]), NOW)
    assert s["assets_ok"] == 0 and len(s["assets_failed"]) >= 4


def test_import_events_csv_and_set_actual(small_world, tmp_path):
    cfg, store, _ = small_world
    p = tmp_path / "e.csv"
    p.write_text("name,currency,datetime_utc,importance,forecast,previous,actual\nCPI y/y,USD,2025-01-15T13:30:00Z,high,2.9%,2.7%,\nBAD,USD,notadate,high,,,\n")
    r = pipeline.import_events_csv(store, str(p), NOW)
    assert r == {"imported": 1, "rejected": 1}
    ev = store.events_between(1736947800 - 1, 1736947800 + 1)[0]
    assert ev.actual is None and ev.forecast == 2.9
    assert pipeline.set_actual(store, ev.event_id, "3.0%", NOW) and store.get_event(ev.event_id).actual == 3.0
    assert not pipeline.set_actual(store, ev.event_id, "junk", NOW) and not pipeline.set_actual(store, "nope", "1", NOW)


def test_train_all_and_evaluate_run_on_small_history(small_world):
    cfg, store, _ = small_world
    reps = pipeline.train_all(cfg, store, NOW)
    assert set(reps) == {f"{s}_{h * 60}" for s in ("pre", "post") for h in cfg.horizons}
    assert all(r["status"] in ("NOT_TRAINED", "EXPERIMENTAL", "VALIDATED") for r in reps.values())
    r = pipeline.run_evaluate(cfg, store, None, NOW, False, True, out=lambda *_: None)
    assert "ACCURACY SUMMARY" in r["summary"]
