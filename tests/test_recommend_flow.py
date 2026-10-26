"""Recommendation engine end-to-end on a SYNTHETIC world (your 28 pair names, random prices) with mocked HTTP. No live APIs are touched."""
import copy
import json
import os
import re
import shutil

import pytest

from uamis import alerts, features, pipeline
from uamis.adapters import coingecko
from uamis.adapters.calendar_ff import make_event_id
from uamis.adapters.http import HttpClient, HttpError
from uamis.classify import classify_event
from uamis.config import Config
from uamis.demo import DEMO_PAIRS, make_demo_config, synth_events, synth_market
from uamis.models import Event
from uamis.prices import MarketData, PriceSeries
from uamis.recommend import crypto as rcrypto, cycle, evidence, report, schedule
from uamis.recommend.engine import Engine
from uamis.recommend.models import AWAY, DIRECTIONS, HZ_NAME, LONG, SHORT, WAIT, CurrencyOutlook, EventCtx, Recommendation
from uamis.storage import Store
from uamis.symbols import asset_kind
from tests.conftest import END, NOW
from tests.test_adapters import FakeResp, FakeSession

KEEP = ("XAUUSD", "DXY", "NAS100", "SPX", "BTCUSD", "ETHUSD", "SOLUSD")


@pytest.fixture(scope="module")
def template(tmp_path_factory, ):
    from uamis.config import load_config
    d = tmp_path_factory.mktemp("recworld")
    base = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    cfg = make_demo_config(str(d), pairs=DEMO_PAIRS, base_cfg=base)
    cfg.raw["assets"] = {k: v for k, v in base.raw["assets"].items() if k in KEEP}
    cfg._assets_cache = None
    store = Store(cfg.db_path)
    evs = synth_events(END - 86400, days=45)
    for e in evs:
        store.upsert_event(e)
    synth_market(cfg, store, END, days=45, events=evs)
    store.close()
    return d, cfg.raw


@pytest.fixture
def world(template, tmp_path):
    d, raw = template
    shutil.copy(os.path.join(str(d), "uamis.sqlite"), tmp_path / "uamis.sqlite")
    r = copy.deepcopy(raw)
    r["db_path"], r["model_dir"] = str(tmp_path / "uamis.sqlite"), str(tmp_path / "models")
    cfg = Config(r)
    store = Store(cfg.db_path)
    yield cfg, store
    store.close()


def fmt(v):
    return "" if v is None else f"{v:g}"


def add_event(store, name, ccy, t0, actual=None, forecast=None, previous=None, importance=3):
    cat, _ = classify_event(name)
    e = Event(make_event_id(ccy, name, t0), name, ccy, t0, importance, cat, fmt(forecast), fmt(previous), fmt(actual), forecast, previous, actual,
              "test", NOW, NOW, t0 + 60 if actual is not None else None)
    store.upsert_event(e)
    return e


def clear_recent(store, days=6):
    """Remove the synthetic history's recent releases so macro-direction tests are deterministic."""
    store.db.execute("DELETE FROM events WHERE t0>?", (NOW - days * 86400,)); store.db.commit()


def run_engine(cfg, store, now=NOW, hist_post=None, det=False, cx=None):
    return Engine(cfg, store, now, pipeline.load_market(cfg, store), hist_post, det, cx).run()


class Snd:
    configured = True

    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send(self, t, m=3500):
        if self.fail:
            raise alerts.TelegramError("boom")
        self.sent.append(t)


# ============================================================================ structure / horizons / coverage
def test_every_instrument_and_horizon_has_a_structured_recommendation(world):
    cfg, store = world
    res = run_engine(cfg, store)
    assert len(res["recs"]) == 28 + len(KEEP) and set(res["currency"]) == set(evidence.CURRENCIES)
    pairs = [n for n in res["recs"] if asset_kind(cfg.all_assets()[n]) == "forex"]
    assert pairs == DEMO_PAIRS                                   # exactly the 28 supplied pairs, same order, none added/removed/duplicated
    for name, by_h in res["recs"].items():
        assert set(by_h) == {900, 3600, 14400, 86400}
        for h, r in by_h.items():
            assert r.instrument == name and r.horizon_s == h and r.direction in DIRECTIONS and -100 <= r.score <= 100
            assert r.confidence in ("n/a", "LOW", "MODERATE", "HIGH") and r.status and r.freshness and r.stage and r.created_ts == NOW
            assert isinstance(r.supporting, list) and isinstance(r.contradicting, list) and r.risks is not None
            assert r.to_dict()["horizon"] == HZ_NAME[h]


def test_unvalidated_by_default_and_never_high_confidence(world):
    cfg, store = world
    res = run_engine(cfg, store)
    assert not any(res["validation"].values())
    for by_h in res["recs"].values():
        for r in by_h.values():
            assert r.confidence != "HIGH" and r.status != "VALIDATED"


# ============================================================================ released data direction + currency logic
def test_hawkish_usd_release_pushes_the_right_way_per_instrument(world):
    cfg, store = world
    clear_recent(store)
    add_event(store, "Non-Farm Employment Change", "USD", NOW - 1200, actual=260e3, forecast=180e3, previous=170e3)
    res = run_engine(cfg, store)
    assert res["currency"]["USD"].direction == "BULLISH" and res["currency"]["EUR"].direction == "NEUTRAL"
    comp = lambda n: res["recs"][n][14400].components["macro"]
    assert comp("USDJPY") > 0 and comp("USDCHF") > 0                     # USD base: supportive
    assert comp("EURUSD") < 0 and comp("AUDUSD") < 0 and comp("GBPUSD") < 0   # USD quote: against
    assert comp("XAUUSD") < 0 and comp("NAS100") < 0 and comp("BTCUSD") < 0   # textbook inverse for gold / equities / crypto
    assert comp("DXY") > 0
    assert comp("EURGBP") is None and comp("AUDNZD") is None             # no USD leg => USD release is not a driver (pair-specific, not common)


def test_dovish_surprise_reverses_signs(world):
    cfg, store = world
    clear_recent(store)
    add_event(store, "Non-Farm Employment Change", "USD", NOW - 1200, actual=100e3, forecast=180e3, previous=170e3)
    res = run_engine(cfg, store)
    assert res["currency"]["USD"].direction == "BEARISH"
    assert res["recs"]["USDJPY"][14400].components["macro"] < 0 and res["recs"]["EURUSD"][14400].components["macro"] > 0


def test_not_every_positive_surprise_strengthens_a_currency(world):
    cfg, store = world       # unemployment ABOVE forecast is dovish for USD (indicator-specific interpretation)
    clear_recent(store)
    add_event(store, "Unemployment Rate", "USD", NOW - 1200, actual=4.4, forecast=4.1, previous=4.1)
    res = run_engine(cfg, store)
    assert res["currency"]["USD"].direction == "BEARISH" and res["recs"]["EURUSD"][14400].components["macro"] > 0


def test_in_line_result_and_missing_forecast_produce_no_macro_driver(world):
    cfg, store = world
    clear_recent(store)
    add_event(store, "CPI y/y", "USD", NOW - 1200, actual=3.1, forecast=3.1, previous=3.0)
    add_event(store, "Retail Sales m/m", "EUR", NOW - 1500, actual=0.4, forecast=None, previous=0.2)
    res = run_engine(cfg, store)
    assert res["currency"]["USD"].direction == "NEUTRAL" and res["currency"]["EUR"].direction == "NEUTRAL"


def test_surprise_sizing_and_revision_detection_from_own_history(world):
    cfg, store = world
    store.db.execute("DELETE FROM events WHERE name='CPI y/y' AND currency='USD'"); store.db.commit()    # control this indicator's history
    add_event(store, "CPI y/y", "USD", NOW - 40 * 86400, actual=3.0, forecast=3.0, previous=2.9)
    e = add_event(store, "CPI y/y", "USD", NOW - 1200, actual=3.4, forecast=3.1, previous=3.2)     # previous revised 3.0 -> 3.2
    m = evidence.macro_scores(store, cfg, NOW)["USD"]
    text = m["items"][0][1]
    assert "revised 3->3.2" in text and "sign only" in text and "hawkish" in text
    for i in range(1, 10):                                     # enough history -> sized in sigma
        add_event(store, "CPI y/y", "USD", NOW - (60 + i * 30) * 86400, actual=3.0 + (i % 3) * 0.1, forecast=3.0, previous=3.0)
    assert "sigma" in evidence.macro_scores(store, cfg, NOW)["USD"]["items"][0][1]
    assert e.currency == "USD"


def test_implausible_surprise_size_falls_back_to_sign_only(world):
    cfg, store = world
    clear_recent(store)
    store.db.execute("DELETE FROM events WHERE name='Retail Sales m/m' AND currency='USD'"); store.db.commit()
    for i in range(10):                                              # tiny, consistent past surprises (std ~0.05)
        add_event(store, "Retail Sales m/m", "USD", NOW - (20 + i) * 86400, actual=0.3 + (0.05 if i % 2 else -0.05), forecast=0.3, previous=0.3)
    add_event(store, "Retail Sales m/m", "USD", NOW - 1200, actual=300.0, forecast=0.3, previous=0.3)       # wildly out of scale (unit mismatch)
    item = evidence.macro_scores(store, cfg, NOW)["USD"]["items"][0]
    assert "implausible" in item[1] and "sign only" in item[1] and abs(item[0]) < 0.40     # sign-only (0.5) + small revision, NOT the ~0.49 a saturated z-score would give


def test_old_releases_decay_and_expire(world):
    cfg, store = world
    clear_recent(store)
    add_event(store, "Non-Farm Employment Change", "USD", NOW - 20 * 3600, actual=260e3, forecast=180e3, previous=170e3)
    fresh = evidence.macro_scores(store, cfg, NOW - 19 * 3600 + 7200)["USD"]["score"]
    later = evidence.macro_scores(store, cfg, NOW)["USD"]["score"]
    assert fresh > later > 0
    assert evidence.macro_scores(store, cfg, NOW + 10 * 86400)["USD"]["score"] is None      # outside lookback


# ============================================================================ pre-news / post-news handling in the engine
def test_pre_news_blackout_two_sided_wait_and_unaffected_pairs(world):
    cfg, store = world
    add_event(store, "CPI y/y", "USD", NOW + 600, forecast=3.1, previous=3.0)            # 10 min away
    r = run_engine(cfg, store)["recs"]
    assert r["EURUSD"][900].direction == AWAY and r["EURUSD"][3600].status == "EVENT RISK" and r["EURUSD"][900].stage == "PRE-NEWS BIAS"
    assert r["EURUSD"][86400].status != "EVENT RISK"                                         # long horizons are not blacked out
    assert r["EURGBP"][3600].status != "EVENT RISK" and r["EURGBP"][3600].stage == "SCAN"    # no USD leg => unaffected
    assert r["XAUUSD"][3600].status == "EVENT RISK" and r["BTCUSD"][3600].status == "EVENT RISK"   # USD-quoted assets are affected
    store.db.execute("DELETE FROM events WHERE name='CPI y/y' AND t0>?", (NOW,)); store.db.commit()
    add_event(store, "CPI y/y", "USD", NOW + 3600, forecast=3.1, previous=3.0)           # 60 min away: two-sided WAIT, not blackout
    r = run_engine(cfg, store)["recs"]
    assert r["EURUSD"][3600].direction in (WAIT, AWAY) and r["EURUSD"][3600].status in ("EVENT RISK", "EXTREME VOLATILITY", "MARKET CLOSED")
    assert r["EURUSD"][3600].event == "USD CPI y/y"


def test_non_usd_event_affects_only_its_own_currency(world):
    cfg, store = world
    add_event(store, "Official Bank Rate", "JPY", NOW + 600, forecast=0.5, previous=0.5)
    r = run_engine(cfg, store)["recs"]
    assert r["USDJPY"][3600].status == "EVENT RISK" and r["EURJPY"][3600].status == "EVENT RISK"
    assert r["EURUSD"][3600].status != "EVENT RISK" and r["XAUUSD"][3600].status != "EVENT RISK"


def test_post_release_spike_window_then_pending_when_actual_missing(world):
    cfg, store = world
    add_event(store, "Non-Farm Employment Change", "USD", NOW - 300, actual=260e3, forecast=180e3, previous=170e3)   # 5 min ago
    r = run_engine(cfg, store)["recs"]
    assert r["EURUSD"][3600].status == "EVENT RISK" and r["EURUSD"][3600].stage == "POST-NEWS" and r["EURUSD"][14400].status != "EVENT RISK"
    store.db.execute("DELETE FROM events"); store.db.commit()
    add_event(store, "CPI y/y", "USD", NOW - 1200, actual=None, forecast=3.1, previous=3.0)       # released 20 min ago, actual unavailable
    r = run_engine(cfg, store)["recs"]
    for h in (900, 3600, 14400, 86400):
        x = r["EURUSD"][h]
        assert x.direction == WAIT and x.status == "PENDING" and x.stage == "POST-NEWS PENDING" and "pending" in x.risks[0]
    assert r["EURGBP"][3600].status != "PENDING"


# ============================================================================ data quality: stale / missing / closed
def test_stale_missing_and_weekend_handling(world):
    cfg, store = world
    mds = pipeline.load_market(cfg, store)
    del mds["BTCUSD"]
    old = mds["XAUUSD"].series[3600]
    mds["XAUUSD"] = MarketData({3600: PriceSeries(3600, [(int(t), 1, 1, 1, c, 0) for t, c in zip(old.ts[:-300], old.c[:-300])])})   # data ends days ago
    r = Engine(cfg, store, NOW, mds).run()["recs"]
    assert r["BTCUSD"][3600].direction == AWAY and r["BTCUSD"][3600].status == "UNAVAILABLE" and r["BTCUSD"][3600].freshness == "UNAVAILABLE"
    assert r["XAUUSD"][3600].direction == AWAY and r["XAUUSD"][3600].status in ("STALE", "MARKET CLOSED", "UNAVAILABLE")
    sat = NOW + 86400                                    # Saturday: FX closed, crypto still 24/7 but our data is a day old
    rs = run_engine(cfg, store, now=sat)["recs"]
    assert rs["EURUSD"][3600].status == "MARKET CLOSED" and rs["EURUSD"][3600].direction == WAIT and rs["EURUSD"][3600].freshness == "CLOSED"
    assert rs["ETHUSD"][3600].status == "STALE" and rs["ETHUSD"][3600].direction == AWAY


# ============================================================================ model / history evidence uses the real components
def _pred(store, asset, h, label, up, down, eid="e1"):
    from uamis.models import Prediction
    p = Prediction(f"p_{asset}_{h}_{label}", eid, asset, "post", h, NOW - 300, NOW - 300, {"UP": up, "DOWN": down, "SIDEWAYS": 1 - up - down},
                   {"UP": .33, "DOWN": .33, "SIDEWAYS": .34}, "UP", label, "v", 30, {}, "LOW", "OK", [], {}, 0.001, .5, "NFP")
    store.insert_prediction(p)


def test_validated_model_counts_fully_experimental_is_downweighted_deterioration_excludes(world):
    cfg, store = world
    _pred(store, "XAUUSD", 3600, "VALIDATED", .6, .2)
    eng = Engine(cfg, store, NOW, pipeline.load_market(cfg, store))
    meta = cfg.all_assets()["XAUUSD"]
    m = [e for e in eng.evidence_for("XAUUSD", meta, "metal", 3600) if e.key == "model"][0]
    assert m.value == pytest.approx(.4) and m.weight == pytest.approx(cfg.raw["recommend"]["weights"]["other"][60]["model"]) and "validated ML" in m.text
    assert [e for e in Engine(cfg, store, NOW, pipeline.load_market(cfg, store), None, True).evidence_for("XAUUSD", meta, "metal", 3600) if e.key == "model"][0].value is None
    store.db.execute("DELETE FROM predictions"); store.db.commit()
    _pred(store, "XAUUSD", 3600, "EXPERIMENTAL", .6, .2)
    m2 = [e for e in Engine(cfg, store, NOW, pipeline.load_market(cfg, store)).evidence_for("XAUUSD", meta, "metal", 3600) if e.key == "model"][0]
    assert m2.weight == pytest.approx(m.weight * .25) and "EXPERIMENTAL" in m2.text
    store.db.execute("DELETE FROM predictions"); store.db.commit()
    m3 = [e for e in Engine(cfg, store, NOW, pipeline.load_market(cfg, store)).evidence_for("XAUUSD", meta, "metal", 3600) if e.key == "model"][0]
    assert m3.value is None and "no active" in m3.text                       # nothing is invented when there is no forecast


def test_historical_reaction_evidence_needs_enough_comparable_events(world):
    cfg, store = world
    e = add_event(store, "Non-Farm Employment Change", "USD", NOW - 1200, actual=260e3, forecast=180e3, previous=170e3)
    def samples(n, label):
        return [features.Sample(f"h{i}", "XAUUSD", "post", 3600, 0, 0, 100, "NFP", {"surprise_sign": 1.0}, .001, .0, label) for i in range(n)]
    eng = lambda hp: Engine(cfg, store, NOW, pipeline.load_market(cfg, store), hp)
    meta = cfg.all_assets()["XAUUSD"]
    pick = lambda en: [x for x in en.evidence_for("XAUUSD", meta, "metal", 3600) if x.key == "reaction"][0]
    ok = pick(eng({("post", 3600): samples(25, "UP")}))
    assert ok.value is not None and ok.value > 0.5 and "n=25" in ok.text and "ABOVE" in ok.text
    thin = pick(eng({("post", 3600): samples(5, "UP")}))
    assert thin.value is None and "INSUFFICIENT HISTORICAL DATA" in thin.text
    assert pick(eng(None)).value is None
    assert e.currency == "USD"


# ============================================================================ crypto
def test_crypto_market_context_and_coin_specific_differences(world):
    cfg, store = world
    mds = pipeline.load_market(cfg, store)
    g = {"btc_dominance": 55.0, "mcap_change_24h_pct": 6.0, "ts": NOW}
    store.kv_set(rcrypto.KV, json.dumps([[NOW - 86400, 54.0]]))
    cx = rcrypto.CryptoContext(cfg, store, mds, NOW, g)
    assert cx.dom_delta == pytest.approx(1.0)
    v, text, _ = cx.market(86400)
    assert v is not None and v > 0 and "total mcap 24h +6.0%" in text and "BTC.D 55.0%" in text and "breadth" in text
    adj_btc, _ = cx.alt_adjust("BTC"); adj_sol, why = cx.alt_adjust("SOL")
    assert adj_btc == 0.0 and adj_sol < 0 and "dominance" in why                 # rising BTC dominance = relative pressure on alts only
    res = run_engine(cfg, store, cx=cx)["recs"]
    btc, sol = res["BTCUSD"][14400].components, res["SOLUSD"][14400].components
    assert btc["cryptomkt"] > sol["cryptomkt"]
    none = rcrypto.CryptoContext(cfg, store, mds, NOW, None)
    assert none.market(86400)[1].startswith("crypto market:")                     # breadth-only fallback, no market-cap claim
    assert "mcap" not in none.market(86400)[1]


def test_crypto_context_unavailable_is_reported_not_invented(world):
    cfg, store = world
    mds = {k: v for k, v in pipeline.load_market(cfg, store).items() if k not in ("ETHUSD", "SOLUSD")}
    cx = rcrypto.CryptoContext(cfg, store, mds, NOW, None)
    v, text, _ = cx.market(3600)
    assert v is None and "unavailable" in text                                   # one coin is not a 'market'


def test_coingecko_parse_and_failure_isolation(world):
    cfg, store = world
    payload = {"data": {"market_cap_percentage": {"btc": 52.3}, "market_cap_change_percentage_24h_usd": -1.4, "total_market_cap": {"usd": 2.5e12}}}
    assert coingecko.parse_global(payload) == {"btc_dominance": 52.3, "mcap_change_24h_pct": -1.4, "total_mcap_usd": 2.5e12}
    assert coingecko.parse_global({"data": {}}) is None and coingecko.parse_global(None) is None
    class Down:
        def get_json(self, *a, **k): raise HttpError("down")
    assert rcrypto.fetch_global(cfg, store, Down(), NOW) is None
    assert rcrypto.fetch_global(cfg, store, None, NOW) is None
    assert run_engine(cfg, store)["recs"]["BTCUSD"][3600].instrument == "BTCUSD"       # engine unaffected by the outage


def test_api_rate_limit_is_honoured_for_coingecko(world):
    cfg, store = world
    payload = {"data": {"market_cap_percentage": {"btc": 52.3}, "market_cap_change_percentage_24h_usd": 1.0}}
    sleeps = []
    sess = FakeSession([FakeResp(429, {}, {"Retry-After": "7"}), FakeResp(200, payload)])
    http = HttpClient(retries=2, backoff=1.0, min_interval=0, session=sess, sleep=sleeps.append)
    g = rcrypto.fetch_global(cfg, store, http, NOW)
    assert g["btc_dominance"] == 52.3 and 7.0 in sleeps and sess.calls == 2
    assert json.loads(store.kv_get(rcrypto.KV))[-1][1] == 52.3                        # dominance snapshot recorded for later 24h deltas
    sess2 = FakeSession([FakeResp(429, {})] * 3)
    assert rcrypto.fetch_global(cfg, store, HttpClient(retries=2, backoff=.1, min_interval=0, session=sess2, sleep=lambda s: None), NOW) is None


# ============================================================================ report building / grouping
def fake_result(cfg, now=NOW, dirs=None, events=(), crypto_text="crypto market: breadth 2/3 coins up", cur=None, comps=None):
    dirs, comps = dirs or {}, comps or {}
    recs = {}
    for n, meta in cfg.all_assets().items():
        recs[n] = {}
        for h in (900, 3600, 14400, 86400):
            d = dirs.get((n, h), WAIT)
            recs[n][h] = Recommendation(n, h, d, 40.0 if d == LONG else -40.0 if d == SHORT else 0.0, "MODERATE" if d in (LONG, SHORT) else "n/a",
                                        "UNVALIDATED" if d in (LONG, SHORT) else ("STALE" if d == AWAY else "NO CLEAR PREFERENCE"), [], [], [], now - 60, "RECENT",
                                        None, "SCAN", now, 1.0, "", asset_kind(meta), 0.001, .5, False, comps.get((n, h), {}))
    c = {x: CurrencyOutlook(x, 0.0, "NEUTRAL", "n/a", ["no recent released data / tone for this currency"], None, None) for x in evidence.CURRENCIES}
    c.update(cur or {})
    return {"recs": recs, "currency": c, "events": list(events), "now": now, "crypto": crypto_text, "validation": {HZ_NAME[h]: False for h in (900, 3600, 14400, 86400)}}


def due_for(store, name, ccy, t0, kind, **kw):
    e = add_event(store, name, ccy, t0, **kw)
    return schedule.DueItem(e, "pre_60" if kind == "pre" else "post_15", kind)


def test_every_pair_is_listed_exactly_once_in_grouped_sections(world):
    cfg, store = world
    text = report.build_report(cfg, fake_result(cfg), [], NOW)
    forex_block = text[text.index("FOREX"):text.index("OTHER MARKETS")]
    for p in DEMO_PAIRS:
        assert len(re.findall(rf"\b{p}\b", forex_block)) == 1, p
    assert {report.group_of(p) for p in DEMO_PAIRS} <= set(report.GROUP_ORDER)
    expected = {"EURGBP": "EUR", "USDJPY": "USD", "EURJPY": "EUR", "GBPJPY": "GBP", "CHFJPY": "JPY", "AUDJPY": "JPY", "AUDNZD": "AUD",
                "NZDCAD": "NZD", "CADCHF": "CAD", "EURCHF": "EUR", "GBPNZD": "GBP", "AUDCHF": "AUD", "NZDCHF": "NZD"}
    assert all(report.group_of(p) == g for p, g in expected.items())


def test_identical_wait_groups_are_compressed_to_one_line(world):
    cfg, store = world
    text = report.build_report(cfg, fake_result(cfg), [], NOW)
    assert "USD PAIRS: EURUSD, GBPUSD, USDJPY" in text and "all WAIT (1h)" in text
    assert text.count("\n") < 60 and len(text) < 3500                          # concise: one message, not dozens of long ones


def test_related_pairs_are_not_forced_into_the_same_direction(world):
    cfg, store = world
    dirs = {("EURUSD", 3600): SHORT, ("GBPUSD", 3600): WAIT, ("USDJPY", 3600): LONG, ("AUDUSD", 3600): SHORT, ("USDCAD", 3600): AWAY}
    cur = {"USD": CurrencyOutlook("USD", 38.0, "BULLISH", "LOW", ["USD Non-Farm Employment Change: actual 260K vs forecast 180K -> hawkish"], 0.4, NOW)}
    text = report.build_report(cfg, fake_result(cfg, dirs=dirs, cur=cur), [], NOW)
    assert re.search(r"EURUSD — SHORT PREFERENCE \(1h, MOD\)", text) and re.search(r"USDJPY — LONG PREFERENCE", text)
    assert re.search(r"GBPUSD — WAIT", text) and re.search(r"USDCAD — STAY AWAY", text)
    assert "USD: BULLISH (+38, LOW)" in text and "SHORT EURUSD" in text and "LONG USDJPY" in text and "USD-bullish expressions" in text


def test_consolidated_pre_news_report_has_required_sections_and_labels(world):
    cfg, store = world
    cfg.raw["recommend"]["display_timezone"] = "Asia/Karachi"
    t0 = NOW + 3600
    d = due_for(store, "CPI y/y", "USD", t0, "pre", forecast=3.1, previous=3.0)
    ev = [EventCtx(d.event.event_id, "CPI y/y", "USD", t0, 3, "pre", 60.0, "3.1", "3.0", "", [p for p in DEMO_PAIRS if "USD" in (p[:3], p[3:])])]
    text = report.build_report(cfg, fake_result(cfg, events=ev), [d], NOW)
    for must in ("UAMIS | MARKET DIRECTION REPORT", "Analysis: PRE-NEWS BIAS (not a confirmed prediction)", "Event: USD CPI y/y", "Actual: not released", "Forecast 3.1",
                 "PKT)", "Updated:", "CURRENCY OUTLOOK", "FOREX", "OTHER MARKETS", "CRYPTO OUTLOOK", "TIMEFRAME SUMMARY", "15m:", "1h:", "4h:", "24h:",
                 "OVERALL PREFERENCE", "RISK: MODERATE", "SCENARIOS (textbook theory, not a forecast)", "UAMIS does not trade", "UNVALIDATED"):
        assert must in text, must
    assert "None" not in text and not re.search(r"\bnan\b", text.lower())


def test_post_news_and_pending_reports(world):
    cfg, store = world
    post = due_for(store, "Non-Farm Employment Change", "USD", NOW - 1200, "post", actual=260e3, forecast=180e3, previous=170e3)
    t = report.build_report(cfg, fake_result(cfg), [post], NOW)
    assert "Analysis: POST-NEWS" in t and "Actual 260000" in t and "Forecast 180000" in t and "Previous 170000" in t
    pend = due_for(store, "CPI y/y", "USD", NOW - 1200, "pending", forecast=3.1, previous=3.0)
    pend.kind = "pending"
    t2 = report.build_report(cfg, fake_result(cfg), [pend], NOW)
    assert "POST-NEWS PENDING (actual not yet available)" in t2 and "Actual: not released" in t2


def test_crypto_divergence_and_horizon_conflicts_are_reported(world):
    cfg, store = world
    comps = {(n, 3600): {"cryptomkt": 0.6, "price": -0.5} for n in ("BTCUSD", "ETHUSD", "SOLUSD")}
    comps[("BTCUSD", 3600)] = {"cryptomkt": 0.6, "price": 0.5}
    dirs = {("EURUSD", 3600): SHORT, ("EURUSD", 86400): LONG, ("BTCUSD", 3600): LONG}
    t = report.build_report(cfg, fake_result(cfg, dirs=dirs, comps=comps), [], NOW)
    assert "Broad crypto market: LEAN LONG" in t
    assert re.search(r"ETHUSD.*own price trend diverges", t) and not re.search(r"BTCUSD.*own price trend diverges", t)
    assert "Horizon conflicts: EURUSD (1h S vs 24h L)" in t


def test_report_labels_snapshot_versus_change_triggered_scan(world):
    cfg, store = world
    assert "Analysis: MARKET SNAPSHOT (no scheduled release due)" in report.build_report(cfg, fake_result(cfg), [], NOW)
    t = report.build_report(cfg, fake_result(cfg), [], NOW, scan_changes=["directional classes changed vs previous scan report"])
    assert "Analysis: SCHEDULED SCAN (meaningful directional change)" in t and "Changes: directional classes changed" in t


def test_unavailable_horizons_are_marked_not_guessed(world):
    cfg, store = world
    res = fake_result(cfg)
    for n in res["recs"]:
        res["recs"][n][900].status = "UNAVAILABLE"
    t = report.build_report(cfg, res, [], NOW)
    assert "15m: unavailable" in t and "15m -" in t


# ============================================================================ cycle: scheduling, catch-up, duplicates, failures
def cyc(cfg, store, snd, now, **kw):
    return cycle.run_cycle(cfg, store, None, snd, now, out=lambda *_: None, **kw)


def test_pre_news_checks_fire_once_per_stage_and_catch_up_to_the_latest(world):
    cfg, store = world
    cfg.raw["recommend"]["scan"]["enabled"] = False
    t0 = NOW + 80 * 60
    add_event(store, "CPI y/y", "USD", t0, forecast=3.1, previous=3.0)
    snd = Snd()
    assert cyc(cfg, store, snd, NOW)["stages"] == ["pre_90"] and len(snd.sent) == 1
    assert cyc(cfg, store, snd, NOW + 120)["mode"] == "none" and len(snd.sent) == 1                  # same stage never repeats
    assert cyc(cfg, store, snd, t0 - 55 * 60)["stages"] == ["pre_60"] and len(snd.sent) == 2
    assert cyc(cfg, store, snd, t0 - 10 * 60)["stages"] == ["pre_15"] and len(snd.sent) == 3
    assert cyc(cfg, store, snd, t0 - 9 * 60)["mode"] == "none" and len(snd.sent) == 3
    assert all("PRE-NEWS BIAS (not a confirmed prediction)" in m for m in snd.sent)


def test_late_scheduler_run_reports_only_the_latest_due_stage(world):
    cfg, store = world
    cfg.raw["recommend"]["scan"]["enabled"] = False
    t0 = NOW + 10 * 60                                   # first run is only 10 min before release: pre_90, pre_60, pre_15 are all 'due'
    add_event(store, "CPI y/y", "USD", t0, forecast=3.1, previous=3.0)
    snd = Snd()
    r = cyc(cfg, store, snd, NOW)
    assert r["stages"] == ["pre_15"] and len(snd.sent) == 1                                          # one message, not three
    assert store.kv_get(schedule.proc_key(make_event_id("USD", "CPI y/y", t0), "pre_90")) is not None   # earlier stages marked superseded
    assert cyc(cfg, store, snd, NOW + 60)["mode"] == "none" and len(snd.sent) == 1


def test_post_news_checks_at_15_and_60_minutes_then_stop(world):
    cfg, store = world
    cfg.raw["recommend"]["scan"]["enabled"] = False
    t0 = NOW - 20 * 60
    add_event(store, "Non-Farm Employment Change", "USD", t0, actual=260e3, forecast=180e3, previous=170e3)
    snd = Snd()
    assert cyc(cfg, store, snd, NOW)["stages"] == ["post_15"] and len(snd.sent) == 1
    assert cyc(cfg, store, snd, NOW + 600)["mode"] == "none"
    assert cyc(cfg, store, snd, t0 + 65 * 60)["stages"] == ["post_60"] and len(snd.sent) == 2
    assert cyc(cfg, store, snd, t0 + 6 * 3600)["mode"] == "none" and len(snd.sent) == 2             # window closed
    assert all("POST-NEWS" in m for m in snd.sent)


def test_pending_notice_is_sent_once_then_real_post_news_when_actual_arrives(world):
    cfg, store = world
    cfg.raw["recommend"]["scan"]["enabled"] = False
    t0 = NOW - 20 * 60
    e = add_event(store, "CPI y/y", "USD", t0, forecast=3.1, previous=3.0)
    snd = Snd()
    assert cyc(cfg, store, snd, NOW)["stages"] == ["post_pending"] and "POST-NEWS PENDING" in snd.sent[0]
    assert cyc(cfg, store, snd, NOW + 300)["mode"] == "none" and len(snd.sent) == 1                  # not repeated every scan
    assert pipeline.set_actual(store, e.event_id, "3.5", NOW + 400)
    r = cyc(cfg, store, snd, NOW + 600)
    assert r["stages"] == ["post_15"] and len(snd.sent) == 2 and "Actual 3.5" in snd.sent[1] and "PENDING" not in snd.sent[1]


def test_restart_safety_and_failed_send_is_retried_not_lost(world):
    cfg, store = world
    cfg.raw["recommend"]["scan"]["enabled"] = False
    t0 = NOW + 30 * 60
    add_event(store, "CPI y/y", "USD", t0, forecast=3.1, previous=3.0)
    with pytest.raises(alerts.TelegramError):
        cyc(cfg, store, Snd(fail=True), NOW)                                          # delivery failure: nothing may be marked processed
    assert store.db.execute("SELECT COUNT(*) c FROM kv WHERE key LIKE 'rec_stage:%'").fetchone()["c"] == 0
    snd = Snd()
    assert cyc(cfg, store, snd, NOW + 60)["sent"] is True and len(snd.sent) == 1     # next run retries
    store2 = Store(cfg.db_path)                                                       # 'workflow restart': fresh process, same state file
    assert cyc(cfg, store2, snd, NOW + 120)["mode"] == "none" and len(snd.sent) == 1
    store2.close()


def test_dry_run_prints_but_records_nothing(world):
    cfg, store = world
    cfg.raw["recommend"]["scan"]["enabled"] = False
    add_event(store, "CPI y/y", "USD", NOW + 30 * 60, forecast=3.1, previous=3.0)
    printed = []
    r = cycle.run_cycle(cfg, store, None, Snd(), NOW, dry_run=True, out=printed.append)
    assert r["result"] == "dry_run" and printed and "MARKET DIRECTION REPORT" in printed[0]
    assert store.db.execute("SELECT COUNT(*) c FROM kv WHERE key LIKE 'rec_stage:%'").fetchone()["c"] == 0
    assert store.db.execute("SELECT COUNT(*) c FROM recommendations").fetchone()["c"] == 0
    snd = Snd()
    assert cyc(cfg, store, snd, NOW + 60)["sent"] is True and len(snd.sent) == 1


def test_sent_reports_log_only_directional_preferences_for_later_evaluation(world, monkeypatch):
    cfg, store = world
    dirs = {("EURUSD", 3600): SHORT, ("USDJPY", 14400): LONG, ("BTCUSD", 86400): LONG, ("GBPUSD", 3600): AWAY}
    monkeypatch.setattr(cycle, "compute", lambda *a, **k: fake_result(cfg, dirs=dirs))
    snd = Snd()
    r = cyc(cfg, store, snd, NOW)
    assert r["sent"] and r["stored"] == 3 and r["n_directional"] == 3
    rows = store.db.execute("SELECT instrument,horizon_s,direction,thr FROM recommendations ORDER BY instrument").fetchall()
    assert [(x["instrument"], x["horizon_s"], x["direction"]) for x in rows] == [("BTCUSD", 86400, LONG), ("EURUSD", 3600, SHORT), ("USDJPY", 14400, LONG)]
    assert all(x["thr"] for x in rows)
    assert cyc(cfg, store, snd, NOW + 4000)["result"] == "duplicate" and store.db.execute("SELECT COUNT(*) c FROM recommendations").fetchone()["c"] == 3


def test_scan_reports_only_on_meaningful_directional_change(world, monkeypatch):
    cfg, store = world
    state = {"dirs": {}}
    monkeypatch.setattr(cycle, "compute", lambda *a, **k: fake_result(cfg, dirs=dict(state["dirs"])))
    snd = Snd()
    assert cyc(cfg, store, snd, NOW)["result"] == "skipped_no_preference" and not snd.sent        # all-WAIT first scan: nothing to say
    state["dirs"] = {("EURUSD", 3600): SHORT, ("EURUSD", 14400): SHORT}
    assert cyc(cfg, store, snd, NOW + 3700)["sent"] and len(snd.sent) == 1                          # first real preference
    assert cyc(cfg, store, snd, NOW + 7400)["result"] == "duplicate"                                # unchanged
    state["dirs"][("EURUSD", 900)] = LONG                                                           # 15m is display-only for change detection
    assert cyc(cfg, store, snd, NOW + 11100)["result"] == "duplicate"
    state["dirs"][("GBPUSD", 3600)] = AWAY                                                          # WAIT -> STAY AWAY is the same 'no preference' class
    assert cyc(cfg, store, snd, NOW + 14800)["result"] == "duplicate"
    state["dirs"][("EURUSD", 3600)] = LONG                                                          # SHORT -> LONG: a real change
    r = cyc(cfg, store, snd, NOW + 14900)
    assert r["sent"] and len(snd.sent) == 2 and "Changes:" in snd.sent[1]
    state["dirs"][("EURUSD", 3600)] = SHORT                                                         # flips straight back
    assert cyc(cfg, store, snd, NOW + 14960)["result"] == "cooldown" and len(snd.sent) == 2         # minimum interval stops flapping messages
    assert cyc(cfg, store, snd, NOW + 14900 + 3700)["sent"] and len(snd.sent) == 3


def test_event_report_becomes_the_scan_baseline_no_immediate_repeat(world, monkeypatch):
    cfg, store = world
    dirs = {("XAUUSD", 14400): SHORT}
    monkeypatch.setattr(cycle, "compute", lambda *a, **k: fake_result(cfg, dirs=dirs))
    add_event(store, "CPI y/y", "USD", NOW + 20 * 60, forecast=3.1, previous=3.0)
    snd = Snd()
    assert cyc(cfg, store, snd, NOW)["mode"] == "event" and len(snd.sent) == 1
    assert cyc(cfg, store, snd, NOW + 120)["result"] == "duplicate" and len(snd.sent) == 1          # same picture is not re-sent as a scan


def test_export_files_for_po3_handoff_are_read_only_labels(world):
    cfg, store = world
    cfg.raw["recommend"]["scan"]["enabled"] = True
    cyc(cfg, store, Snd(), NOW)
    path = os.path.join(os.path.dirname(cfg.db_path), "reports", "recommendations_latest.json")
    data = json.load(open(path))
    assert data["generated_at"] == NOW and "never trades" in data["note"] and len(data["recs"]) == 28 + len(KEEP)
    assert data["recs"]["EURUSD"]["3600"]["horizon"] == "1h" and "direction" in data["recs"]["EURUSD"]["3600"]
    from uamis.recommend import po3
    r = po3.check(store, "EURUSD", "LONG", NOW + 60)
    assert r["alignment"] in (po3.ALIGNED, po3.CONFLICTING, po3.NO_PREF, po3.AWAY_R) and "EURUSD" == r["instrument"]


def test_disabled_flag_turns_the_whole_cycle_off(world):
    cfg, store = world
    cfg.raw["recommend"]["enabled"] = False
    snd = Snd()
    assert cyc(cfg, store, snd, NOW) == {"enabled": False} and not snd.sent


# ============================================================================ integration with the existing tick
class TickHttp:
    def __init__(self, cal):
        self.cal = cal

    def get_json(self, url, **kw):
        if "faireconomy" in url:
            return self.cal
        raise HttpError("price/global API unavailable (simulated)")

    def get_text(self, url, **kw):
        raise HttpError("rss unavailable (simulated)")

    def request(self, *a, **k):
        raise HttpError("n/a")


def iso(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def test_tick_sends_one_consolidated_message_and_keeps_logging_predictions(world):
    cfg, store = world
    feed = [{"title": "CPI y/y", "country": "USD", "date": iso(NOW + 30 * 60), "impact": "High", "forecast": "3.1%", "previous": "3.0%", "actual": ""}]
    snd = Snd()
    out = []
    r = pipeline.run_tick(cfg, store, TickHttp(feed), snd, NOW, out=out.append)
    pre_msgs = [m for m in snd.sent if "MARKET DIRECTION REPORT" in m]
    assert len(pre_msgs) == 1 and r["recommend"]["mode"] == "event" and r["pre"] == 0           # legacy probability message is folded in, not duplicated
    assert not any("Stage: PRE-NEWS" in m for m in snd.sent)
    assert store.db.execute("SELECT COUNT(*) c FROM predictions WHERE stage='pre'").fetchone()["c"] > 0       # ledger logging unchanged
    n = len(snd.sent)
    pipeline.run_tick(cfg, store, TickHttp(feed), snd, NOW + 180, out=out.append)
    assert len(snd.sent) == n                                                                    # no repeat on the next scan


def test_legacy_messages_can_still_be_enabled_alongside(world):
    cfg, store = world
    cfg.raw["recommend"]["legacy_event_reports"] = True
    feed = [{"title": "CPI y/y", "country": "USD", "date": iso(NOW + 30 * 60), "impact": "High", "forecast": "3.1%", "previous": "3.0%", "actual": ""}]
    snd = Snd()
    r = pipeline.run_tick(cfg, store, TickHttp(feed), snd, NOW, out=lambda *_: None)
    assert r["pre"] == 1 and any("Stage: PRE-NEWS" in m for m in snd.sent) and any("MARKET DIRECTION REPORT" in m for m in snd.sent)


def test_recommendation_failure_never_breaks_the_rest_of_the_tick(world, monkeypatch):
    cfg, store = world
    import uamis.recommend.cycle as cy
    monkeypatch.setattr(cy, "run_cycle", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("engine bug")))
    feed = [{"title": "CPI y/y", "country": "USD", "date": iso(NOW + 30 * 60), "impact": "High", "forecast": "3.1%", "previous": "3.0%", "actual": ""}]
    r = pipeline.run_tick(cfg, store, TickHttp(feed), Snd(), NOW, out=lambda *_: None)
    assert "RuntimeError" in r["recommend"]["error"] and r["calendar"]["new"] == 1 and r["heads_up"] >= 1


def test_evaluate_command_scores_recommendations_and_reports_honestly(world):
    cfg, store = world
    r = pipeline.run_evaluate(cfg, store, None, NOW, False, True, out=lambda *_: None)
    assert "ACCURACY SUMMARY" in r["summary"] and "No accuracy is claimed" in r["recommendation_summary"]
    assert os.path.exists(os.path.join(os.path.dirname(cfg.db_path), "reports", "recommendation_metrics.json"))
