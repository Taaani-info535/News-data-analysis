"""Recommendation engine: pure scoring/override logic, symbols, timezones, PO3 mapping, validation, safety scan. SYNTHETIC inputs only."""
import json
import os
import re

import pytest

from uamis import symbols
from uamis.recommend import engine, po3, report, validation
from uamis.recommend.models import AWAY, FUNDAMENTAL, LONG, MODEL, PRICE, SHORT, WAIT, Evidence
from uamis.storage import Store
from uamis.util import parse_iso_to_ts
from tests.conftest import NOW

W60 = {"macro": .22, "tone": .06, "rates": .06, "risk": .10, "model": .22, "reaction": .12, "price": .15, "strength": .07}
KIND = {"model": MODEL, "reaction": MODEL, "price": PRICE, "strength": PRICE}


def evs(vals, weights=W60):
    """vals: key -> value (None = applicable but unavailable); keys not given are NOT applicable (dropped)."""
    return [Evidence(k, k, v, weights[k], f"{k} {v}", KIND.get(k, FUNDAMENTAL)) for k, v in vals.items()]


@pytest.fixture
def th(base_cfg):
    return base_cfg.raw["recommend"]["thresholds"]


def D(vals, th, **fl):
    return engine.decide(evs(vals), engine.Flags(**fl), th)


# ---------------------------------------------------------------- bullish / bearish / neutral / conflicting
def test_bullish_evidence_gives_long_preference_with_explanations(th):
    d, score, conf, status, sup, con, risks, st = D({"macro": .8, "tone": .5, "rates": .6, "risk": .5, "model": .6, "reaction": .5, "price": .4, "strength": .3}, th)
    assert d == LONG and score > 25 and status == "UNVALIDATED" and sup and conf in ("LOW", "MODERATE")
    assert any("UNVALIDATED" in r for r in risks)


def test_bearish_is_mirror_image(th):
    d, score, *_ = D({"macro": -.8, "tone": -.5, "rates": -.6, "risk": -.5, "model": -.6, "reaction": -.5, "price": -.4, "strength": -.3}, th)
    assert d == SHORT and score < -25


def test_weak_evidence_is_neutral_wait(th):
    d, score, conf, status, *_ = D({"macro": .1, "tone": -.05, "rates": .0, "risk": .05, "model": .1, "reaction": .0, "price": .1, "strength": .05}, th)
    assert d == WAIT and status == "NO CLEAR PREFERENCE" and conf == "n/a" and abs(score) < 25


def test_conflicting_evidence_overrides_a_decent_score(th):
    d, score, conf, status, sup, con, risks, st = D({"macro": 1, "tone": 1, "rates": 1, "risk": 1, "model": -1, "reaction": -1, "price": 1, "strength": 1}, th)
    assert abs(score) >= 25 and st["conflict"] > .30
    assert d == WAIT and status == "CONFLICTING" and con           # opposing evidence is reported, not hidden


def test_price_momentum_alone_never_creates_a_preference(th):
    d, score, conf, status, sup, con, risks, st = D({"macro": .05, "price": 1.0, "strength": 1.0}, th)
    assert abs(score) >= 25 and d == WAIT and status == "NO FUNDAMENTAL DRIVER"


def test_missing_applicable_evidence_shrinks_toward_neutral_but_na_does_not(th):
    full = engine.combine(evs({"macro": .8, "model": .8}))
    thin = engine.combine(evs({"macro": .8, "model": .8, "risk": None, "reaction": None, "price": None, "tone": None}))
    assert thin["coverage"] < full["coverage"] == 1.0 and abs(thin["score"]) < abs(full["score"])


def test_high_confidence_requires_validation_recent_data_and_broad_agreement(th):
    vals = {k: .9 for k in W60}
    assert D(vals, th, validated=False)[2] == "MODERATE"            # unvalidated is capped
    assert D(vals, th, validated=True)[2] == "HIGH"
    assert D(vals, th, validated=True, freshness="DELAYED")[2] == "LOW"


# ---------------------------------------------------------------- stale / missing / closed / event overrides
@pytest.mark.parametrize("fl,direction,status", [
    (dict(freshness="UNAVAILABLE"), AWAY, "UNAVAILABLE"),
    (dict(freshness="STALE", age_s=20000), AWAY, "STALE"),
    (dict(freshness="CLOSED"), WAIT, "MARKET CLOSED"),
    (dict(blackout_pre="USD CPI in 10 min", short=True), AWAY, "EVENT RISK"),
    (dict(blackout_post="USD CPI released 4 min ago", short=True), AWAY, "EVENT RISK"),
    (dict(pending_post="USD CPI"), WAIT, "PENDING"),
    (dict(pre_event="USD CPI", pre_importance=3, short=True), WAIT, "EVENT RISK"),
    (dict(extreme_vol=True, vol_pct=.97), AWAY, "EXTREME VOLATILITY"),
])
def test_overrides_beat_a_strong_score(th, fl, direction, status):
    d, score, conf, st, *_ = D({k: .9 for k in W60}, th, **fl)
    assert score > 25 and d == direction and st == status and conf == "n/a"


def test_long_horizons_are_not_blacked_out_but_are_flagged_as_pre_news_bias(th):
    d, score, conf, status, sup, con, risks, _ = D({k: .9 for k in W60}, th, pre_event="USD CPI", pre_importance=3, short=False)
    assert d == LONG and conf in ("LOW", "MODERATE") and any("PRE-NEWS BIAS" in r and "not a confirmed prediction" in r for r in risks)


def test_pre_news_needs_a_stronger_score_than_normal(th):
    vals = {k: .35 for k in W60}                       # ~35: enough normally, not enough under the x1.5 pre-news multiplier
    assert D(vals, th)[0] == LONG
    assert D(vals, th, pre_event="USD CPI", pre_importance=2, short=False)[0] == WAIT


def test_insufficient_coverage_waits(th):
    d, score, conf, status, *_ = D({"macro": .9, "model": None, "reaction": None, "risk": None, "price": None, "strength": None, "tone": None, "rates": None}, th)
    assert d == WAIT and status == "INSUFFICIENT EVIDENCE"


# ---------------------------------------------------------------- weights configuration sanity
def test_weights_sum_to_one_for_every_group_and_horizon(base_cfg):
    for grp, by_h in base_cfg.raw["recommend"]["weights"].items():
        assert set(by_h) == {15, 60, 240, 1440}
        for h, w in by_h.items():
            assert abs(sum(w.values()) - 1.0) < 1e-9, (grp, h)


# ---------------------------------------------------------------- symbols
def test_symbol_validation_accepts_shipped_config_and_rejects_malformed(base_cfg):
    assert symbols.validate_all(base_cfg.all_assets()) == {}
    pairs = [n for n, m in base_cfg.all_assets().items() if m.get("class") == "forex"]
    assert len(pairs) == 28
    assert all(symbols.validate_asset(p, base_cfg.all_assets()[p]) == [] for p in pairs)
    bad = {"source": "yahoo", "symbol": "EUR USD=X", "base": "EUR", "quote": "USD", "class": "forex"}
    assert symbols.validate_asset("EURUSD", bad)
    assert symbols.validate_asset("BTCUSD", {"source": "binance", "symbol": "btcusdt", "base": "BTC", "quote": "USD"})
    assert symbols.validate_asset("X", {"source": "binance", "symbol": "BTCEUR", "base": "BTC", "quote": "USD"})


def test_invalid_symbols_are_never_requested(small_world):
    from uamis import pipeline
    cfg, store, _ = small_world
    cfg.raw["assets"]["XAUUSD"] = dict(cfg.raw["assets"]["XAUUSD"], symbol="GC =F")
    cfg._assets_cache = None
    calls = []
    class H:
        def get_json(self, url, **kw):
            calls.append(url); raise RuntimeError("blocked")
    s = pipeline.collect_prices(cfg, store, H(), NOW)
    assert "invalid symbol" in s["assets_failed"]["XAUUSD"] and not any("GC" in c and "%20" in c for c in calls)
    assert all("XAUUSD" not in k or k == "XAUUSD" for k in s["assets_failed"])


def test_asset_kind_inference():
    assert symbols.asset_kind({"base": "BTC"}) == "crypto" and symbols.asset_kind({"base": "XAU"}) == "metal"
    assert symbols.asset_kind({"base": "IDX"}) == "index" and symbols.asset_kind({"class": "forex"}) == "forex"


# ---------------------------------------------------------------- timezone handling
def test_local_label_conversions_and_dst():
    ts = parse_iso_to_ts("2026-01-14T13:30:00+00:00")
    assert report.local_label(ts, "UTC") == "2026-01-14 13:30 UTC"
    assert report.local_label(ts, "Asia/Karachi").endswith("(18:30 PKT)")
    assert "(08:30 EST)" in report.local_label(ts, "America/New_York")
    summer = parse_iso_to_ts("2026-07-15T12:30:00+00:00")
    assert "(08:30 EDT)" in report.local_label(summer, "America/New_York")        # DST shifts local time, UTC stays the anchor
    assert "unavailable" in report.local_label(ts, "Not/AZone")
    assert report.local_label(None, "UTC") == "n/a"


def test_feed_offsets_convert_to_the_same_utc_instant():
    assert parse_iso_to_ts("2026-01-14T08:30:00-05:00") == parse_iso_to_ts("2026-01-14T13:30:00+00:00") == parse_iso_to_ts("2026-01-14T18:30:00+05:00")
    assert parse_iso_to_ts("2026-07-15T08:30:00-04:00") == parse_iso_to_ts("2026-07-15T12:30:00Z")


def test_weekly_market_closure_helper():
    from uamis.recommend.evidence import market_open
    fri_open, fri_close = parse_iso_to_ts("2026-10-23T10:00:00Z"), parse_iso_to_ts("2026-10-23T22:30:00Z")
    sat, sun_early, sun_open = parse_iso_to_ts("2026-10-24T12:00:00Z"), parse_iso_to_ts("2026-10-25T10:00:00Z"), parse_iso_to_ts("2026-10-25T22:00:00Z")
    assert market_open(fri_open, "forex") and not market_open(fri_close, "forex") and not market_open(sat, "forex")
    assert not market_open(sun_early, "forex") and market_open(sun_open, "forex")
    assert all(market_open(t, "crypto") for t in (fri_close, sat, sun_early))     # crypto is 24/7


# ---------------------------------------------------------------- PO3 relationship (read-only labels)
@pytest.mark.parametrize("side,direction,expected", [
    ("LONG", LONG, po3.ALIGNED), ("SHORT", SHORT, po3.ALIGNED), ("LONG", SHORT, po3.CONFLICTING), ("SHORT", LONG, po3.CONFLICTING),
    ("LONG", WAIT, po3.NO_PREF), ("SHORT", WAIT, po3.NO_PREF), ("LONG", AWAY, po3.AWAY_R)])
def test_po3_alignment_matrix(side, direction, expected):
    assert po3.alignment(side, direction) == expected
    assert po3.alignment(side.lower(), direction) == expected


def test_po3_alignment_rejects_garbage_side():
    with pytest.raises(ValueError):
        po3.alignment("BUY", LONG)


def test_po3_check_reads_latest_recommendation_and_flags_staleness():
    st = Store(":memory:")
    assert po3.check(st, "EURUSD", "LONG", NOW)["alignment"] == po3.UNAVAILABLE            # nothing generated yet
    rec = lambda d: {str(h): {"direction": d, "score": 40.0, "confidence": "LOW", "status": "UNVALIDATED"} for h in (900, 3600, 14400, 86400)}
    st.kv_set("rec_latest", json.dumps({"generated_at": NOW - 600, "recs": {"EURUSD": rec(SHORT), "GBPUSD": rec(LONG)}}))
    r = po3.check(st, "eur/usd", "LONG", NOW)
    assert r["alignment"] == po3.CONFLICTING and r["uamis_1h"] == SHORT and "never invalidates" in r["note"]
    assert po3.check(st, "GBPUSD", "LONG", NOW)["alignment"] == po3.ALIGNED
    assert po3.check(st, "XXXYYY", "LONG", NOW)["alignment"] == po3.UNAVAILABLE
    assert po3.check(st, "EURUSD", "LONG", NOW + 6 * 3600)["alignment"] == po3.UNAVAILABLE   # recommendation too old
    st.close()


# ---------------------------------------------------------------- live validation of the recommendations themselves
def _rec_row(i, instrument="EURUSD", direction=LONG, label="UP", ret=0.002, h=3600):
    return {"rec_id": f"r{i}", "created_ts": 1000 + i, "instrument": instrument, "horizon_s": h, "direction": direction, "score": 40.0,
            "confidence": "MODERATE", "status": "UNVALIDATED", "stage": "SCAN", "event_name": None, "thr": 0.001, "vol_pct": 0.5, "grp": "", "components": {}}, label, ret


def _fill(st, n, hit_rate):
    for i in range(n):
        row, _, _ = _rec_row(i)
        st.insert_recommendation(row)
        hit = (i / n) < hit_rate
        st.set_recommendation_outcome(f"r{i}", 0.002 if hit else -0.002, "UP" if hit else "DOWN", "EVALUATED", 5000)


def test_validated_only_with_enough_scored_outcomes_and_a_credible_edge(base_cfg):
    st = Store(":memory:")
    _fill(st, 30, 0.9)
    assert validation.validation_status(st, base_cfg, 3600)["validated"] is False              # great hit rate but far too few samples
    st.close()
    st = Store(":memory:"); _fill(st, 200, 0.5)
    assert validation.validation_status(st, base_cfg, 3600)["validated"] is False              # coin-flip: no edge
    st.close()
    st = Store(":memory:"); _fill(st, 200, 0.75)
    v = validation.validation_status(st, base_cfg, 3600)
    assert v["validated"] is True and v["n"] == 200 and v["stats"]["hit_ci95"][0] > 0.5
    assert validation.validation_status(st, base_cfg, 86400)["validated"] is False             # other horizons are judged separately
    st.close()


def test_net_return_must_be_positive_for_validation(base_cfg):
    st = Store(":memory:")
    for i in range(200):
        row, _, _ = _rec_row(i); st.insert_recommendation(row)
        hit = i < 150
        st.set_recommendation_outcome(f"r{i}", 0.00001 if hit else -0.01, "UP" if hit else "DOWN", "EVALUATED", 5000)   # right often, but loses money net
    assert validation.validation_status(st, base_cfg, 3600)["validated"] is False
    st.close()


def test_pending_recommendations_are_not_scored_before_horizon(small_world):
    cfg, store, _ = small_world
    row, _, _ = _rec_row(1, instrument="XAUUSD")
    row["created_ts"] = NOW - 600
    store.insert_recommendation(row)
    r = validation.score_pending(store, cfg, NOW)
    assert r == {"evaluated": 0, "pending": 1, "no_data": 0}
    old, _, _ = _rec_row(2, instrument="XAUUSD"); old["created_ts"] = NOW - 3 * 86400          # a weekday (synthetic markets have no weekend candles)
    store.insert_recommendation(old)
    assert validation.score_pending(store, cfg, NOW)["evaluated"] == 1
    m = validation.metrics(store, cfg)
    assert m["overall"]["n"] == 1 and "Sample is small" in validation.format_summary(m)
    assert "No accuracy is claimed" in validation.format_summary(validation.metrics(Store(":memory:"), cfg))


# ---------------------------------------------------------------- safety: no trading, no unexpected hosts
SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uamis")


def _py_files():
    for root, _, files in os.walk(SRC):
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def test_no_order_execution_or_broker_code_anywhere():
    banned = ["create_order", "place_order", "submit_order", "order_send", "MetaTrader5", "import mt5", "ccxt", "newOrder", "/api/v3/order",
              "cancel_order", "set_leverage", "OANDA", "alpaca"]
    hits = [(p, b) for p in _py_files() for b in banned if b in open(p, encoding="utf-8").read()]
    assert hits == []


def test_only_known_data_hosts_appear_in_source_and_config():
    allowed = {"api.telegram.org", "query1.finance.yahoo.com", "data-api.binance.vision", "nfs.faireconomy.media", "www.federalreserve.gov",
               "www.bls.gov", "www.ecb.europa.eu", "api.stlouisfed.org", "api.coingecko.com"}
    text = "".join(open(p, encoding="utf-8").read() for p in _py_files()) + open(os.path.join(os.path.dirname(SRC), "config", "config.yaml")).read()
    hosts = {m.group(1) for m in re.finditer(r"https?://([A-Za-z0-9.\-]+)", text)}
    hosts -= {"x-access-token"}
    assert hosts <= allowed, hosts - allowed


def test_no_secrets_in_config_and_all_modules_compile():
    import compileall
    assert compileall.compile_dir(SRC, quiet=1, force=True)
    cfgtext = open(os.path.join(os.path.dirname(SRC), "config", "config.yaml")).read()
    assert not re.search(r"[0-9]{8,10}:[A-Za-z0-9_-]{30,}", cfgtext) and "TELEGRAM_BOT_TOKEN" not in cfgtext
