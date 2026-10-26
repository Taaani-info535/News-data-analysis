import json
import pytest
from uamis.adapters import calendar_ff, prices_yahoo, prices_binance, news_rss, fred
from uamis.adapters.http import HttpClient, HttpError


def test_calendar_parse_keeps_missing_as_none_and_skips_holidays():
    items = [
        {"title": "CPI m/m", "country": "USD", "date": "2024-01-10T08:30:00-05:00", "impact": "High", "forecast": "0.3%", "previous": "0.1%"},
        {"title": "Bank Holiday", "country": "USD", "date": "2024-01-15T00:00:00-05:00", "impact": "Holiday", "forecast": "", "previous": ""},
        {"title": "Broken", "country": "USD", "date": "not a date", "impact": "High"},
        {"title": "Unemployment Rate", "country": "USD", "date": "2024-01-05T08:30:00-05:00", "impact": "High", "forecast": "", "previous": "3.7%", "actual": "3.7%"},
    ]
    evs = calendar_ff.parse_feed(items, "src", now_ts=1)
    assert len(evs) == 2
    cpi = evs[0]
    assert cpi.t0 == 1704893400 - 0 and cpi.actual is None and cpi.forecast == 0.3 and cpi.category == "CPI"
    assert evs[1].forecast is None and evs[1].actual == 3.7      # forecast never invented


def test_event_id_stable_and_case_insensitive():
    assert calendar_ff.make_event_id("USD", "CPI", 1) == calendar_ff.make_event_id("USD", " cpi ", 1)


def test_yahoo_parse_drops_gaps_and_forming_candle():
    now = 1000 + 300 * 3 + 100
    payload = {"chart": {"result": [{"timestamp": [1000, 1300, 1600, 1900], "indicators": {"quote": [{
        "open": [1, None, 3, 4], "high": [1, 2, 3, 4], "low": [1, 2, 3, 4], "close": [1, 2, 3, 4], "volume": [1, 1, 1, 1]}]}}]}}
    rows = prices_yahoo.parse_chart(payload, 300, now)
    assert [r[0] for r in rows] == [1000, 1600]       # None gap dropped; ts=1900 still forming (1900+300 > now)


def test_yahoo_error_payload_raises():
    with pytest.raises(ValueError):
        prices_yahoo.parse_chart({"chart": {"result": None, "error": {"code": "Not Found"}}}, 300, 10**9)


def test_binance_parse_and_malformed():
    now = 10_000
    k = [[1_000_000, "1", "2", "0.5", "1.5", "10"], [9_900_000, "1", "2", "0.5", "1.5", "10"], ["bad"]]
    rows = prices_binance.parse_klines(k, 300, now)
    assert len(rows) == 1 and rows[0][0] == 1000
    with pytest.raises(ValueError):
        prices_binance.parse_klines({"code": -1}, 300, now)


def test_rss_dedup_by_headline_and_requires_timestamp():
    xml = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
    <item><title>FOMC statement</title><link>http://a</link><pubDate>Wed, 10 Jan 2024 19:00:00 GMT</pubDate></item>
    <item><title>No date item</title><link>http://b</link></item></channel></rss>"""
    now = 1704913200 + 600
    a = news_rss.parse_feed_text(xml, "Fed", True, now, 86400)
    b = news_rss.parse_feed_text(xml, "Other", False, now, 86400)
    assert len(a) == 1 and a[0].verified and not b[0].verified and a[0].news_id == b[0].news_id
    assert news_rss.parse_feed_text(xml, "Fed", True, now + 10 * 86400, 86400) == []     # too old


def test_fred_missing_values_skipped():
    rows = fred.parse_observations({"observations": [{"date": "2024-01-02", "value": "."}, {"date": "2024-01-03", "value": "4.1"}]})
    assert len(rows) == 1 and rows[0][4] == 4.1


class FakeResp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._b, self.headers, self.text = status, body, headers or {}, json.dumps(body)

    def json(self):
        if self._b is None:
            raise ValueError("bad json")
        return self._b


class FakeSession:
    def __init__(self, seq):
        self.seq, self.calls, self.headers = list(seq), 0, {}

    def request(self, method, url, **kw):
        self.calls += 1
        r = self.seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_http_retries_then_succeeds_with_backoff_and_retry_after():
    import requests
    sleeps = []
    s = FakeSession([FakeResp(429, {}, {"Retry-After": "3"}), requests.ConnectionError("x"), FakeResp(200, {"ok": 1})])
    h = HttpClient(retries=3, backoff=1.0, min_interval=0, session=s, sleep=sleeps.append)
    assert h.get_json("https://x.test/a") == {"ok": 1}
    assert s.calls == 3 and 3.0 in sleeps


def test_http_gives_up_and_nonretryable():
    s = FakeSession([FakeResp(500, {})] * 3)
    h = HttpClient(retries=2, backoff=0.1, min_interval=0, session=s, sleep=lambda x: None)
    with pytest.raises(HttpError):
        h.get_json("https://x.test/a")
    assert s.calls == 3
    s2 = FakeSession([FakeResp(404, {})])
    with pytest.raises(HttpError):
        HttpClient(retries=3, min_interval=0, session=s2, sleep=lambda x: None).get_json("https://x.test/a")
    assert s2.calls == 1          # 404 is not retried


def test_http_invalid_json_and_rate_limit_spacing():
    h = HttpClient(retries=0, min_interval=0, session=FakeSession([FakeResp(200, None)]), sleep=lambda x: None)
    with pytest.raises(HttpError):
        h.get_json("https://x.test/a")
    waits = []
    h2 = HttpClient(retries=0, min_interval=5.0, session=FakeSession([FakeResp(200, {}), FakeResp(200, {})]), sleep=waits.append)
    h2.get_json("https://same.host/a"); h2.get_json("https://same.host/b")
    assert waits and waits[0] > 0       # second call to same host was throttled
