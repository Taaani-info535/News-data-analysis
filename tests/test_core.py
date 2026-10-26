import math
import pytest
from uamis.util import parse_value, wilson_interval, parse_iso_to_ts, ts_to_iso
from uamis.validation import validate_candles, staleness_seconds
from uamis.classify import classify_event, classify_news, surprise_sign, surprise_rel, theoretical_impact, relevant_assets, hawkish_rule


def test_parse_value_variants():
    assert parse_value("3.2%") == 3.2
    assert parse_value("227K") == 227000
    assert parse_value("-0.1%") == -0.1
    assert parse_value("1.5M") == 1.5e6


@pytest.mark.parametrize("raw", ["", None, "-", "N/A", "abc", "3.2.1%"])
def test_parse_value_never_invents(raw):
    assert parse_value(raw) is None


def test_wilson_bounds():
    lo, hi = wilson_interval(5, 10)
    assert 0 < lo < 0.5 < hi < 1
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_iso_offset_to_utc():
    assert parse_iso_to_ts("2024-01-10T08:30:00-05:00") == parse_iso_to_ts("2024-01-10T13:30:00Z")
    assert ts_to_iso(parse_iso_to_ts("2024-01-10T13:30:00Z")).startswith("2024-01-10 13:30")


def test_validate_candles_rejects_bad_rows():
    now = 10_000_000
    rows = [(now - 3600, 1, 2, 0.5, 1.5, 1), (now - 7200, float("nan"), 2, 1, 1, 1), (now - 10800, 1, 0.9, 1, 1, 1),
            (now + 99999, 1, 1, 1, 1, 1), (now - 3600, 1, 2, 0.5, 1.6, 1), (now - 14400, -1, 1, 1, 1, 1), ("x", 1, 1, 1, 1, 1)]
    clean, rej = validate_candles(rows, 3600, now)
    assert [r[0] for r in clean] == [now - 3600] and clean[0][4] == 1.6 and rej == 5


def test_staleness():
    assert staleness_seconds(None, 300, 1000) is None
    assert staleness_seconds(1000, 300, 2000) == 700


def test_classification():
    assert classify_event("Core CPI m/m")[0] == "CPI"
    assert classify_event("Non-Farm Employment Change")[0] == "NFP"
    assert classify_event("Unemployment Rate") == ("UNEMPLOYMENT", False)
    assert classify_event("Federal Funds Rate")[0] == "FED_RATE"
    assert classify_event("Something Odd")[0] == "OTHER"
    assert classify_news("Minutes of the Federal Open Market Committee") == "FED_MINUTES"


def test_surprise_and_theory():
    assert surprise_sign(3.5, 3.2) == 1 and surprise_sign(3.0, 3.2) == -1 and surprise_sign(3.2, 3.2) == 0
    assert surprise_sign(None, 3.2) is None and surprise_rel(3.5, None, 3.0) is None
    gold = {"base": "XAU", "quote": "USD"}; eur = {"base": "EUR", "quote": "USD"}; usdjpy = {"base": "USD", "quote": "JPY"}
    hk = hawkish_rule("CPI", "CPI y/y")
    assert theoretical_impact("USD", hk, 1, gold)[0] == "BEARISH"
    assert theoretical_impact("USD", hk, 1, eur)[0] == "BEARISH"
    assert theoretical_impact("USD", hk, 1, usdjpy)[0] == "BULLISH"
    # unemployment higher-than-forecast is dovish => USD weak => gold bullish
    assert theoretical_impact("USD", False, 1, gold)[0] == "BULLISH"
    assert theoretical_impact("USD", None, 1, gold)[0] == "N/A"
    assert theoretical_impact("USD", hk, None, gold)[0] == "N/A"
    assert theoretical_impact("CNY", hk, 1, gold)[0] == "N/A"


def test_relevant_assets():
    a = {"EURUSD": {"base": "EUR", "quote": "USD"}, "EURGBP": {"base": "EUR", "quote": "GBP"}, "XAUUSD": {"base": "XAU", "quote": "USD"}}
    assert relevant_assets("USD", a) == ["EURUSD", "XAUUSD"]
    assert relevant_assets("GBP", a) == ["EURGBP"]


def test_post_anchor_waits_until_actual_is_observed_and_minimum_delay_passes():
    from uamis.features import anchors
    from uamis.models import Event
    ev = Event(event_id="e", name="CPI", currency="USD", t0=10_000, importance=3, category="CPI",
              actual=3.2, actual_seen_at=12_000)
    feat_ts, anchor_ts = anchors(ev, "post", pre_lead_s=5_400, post_delay_s=900)
    assert feat_ts == anchor_ts == 12_000  # later observed-actual time wins
    ev.actual_seen_at = 10_200
    feat_ts, anchor_ts = anchors(ev, "post", pre_lead_s=5_400, post_delay_s=900)
    assert feat_ts == anchor_ts == 10_900  # configured 15-minute minimum after release wins
