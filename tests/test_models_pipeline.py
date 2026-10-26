import json
import math
import numpy as np
import pytest

from uamis import features, ml, eventstudy, risk, evaluate, alerts, pipeline, predict
from uamis.models import Event, Prediction
from uamis.prices import PriceSeries, MarketData
from uamis.storage import Store
from tests.conftest import END, NOW


# ---------------------------------------------------------------- price series / leakage
def mk_series(res, n, start=0, step_px=1.0):
    return PriceSeries(res, [(start + i * res, 100 + i * step_px, 101 + i, 99 + i, 100 + i * step_px, 1) for i in range(n)])


def test_close_at_never_returns_future_candle():
    s = mk_series(3600, 10)
    px, ct = s.close_at(3600 * 5 + 1800)       # mid-way through candle idx5 -> only idx4 closed
    assert ct == 3600 * 5 and px == 104
    assert s.close_at(100) is None
    assert s.close_at(3600 * 5 + 1800, max_age=60) is None     # too old => missing, not filled


def test_gap_returns_are_nan_not_interpolated():
    rows = [(0, 1, 1, 1, 100, 1), (3600, 1, 1, 1, 101, 1), (3600 * 5, 1, 1, 1, 105, 1)]
    r = PriceSeries(3600, rows).hourly_returns()
    assert np.isnan(r[0]) and np.isfinite(r[1]) and np.isnan(r[2])


def test_features_ignore_future_prices(small_world):
    cfg, store, evs = small_world
    ev = evs[40]
    md = MarketData.from_store(store, "XAUUSD")
    ft = ev.t0 - 3600
    f1 = features.build_features(md, ev, "pre", ft, cfg.raw["labels"])
    # tamper with every candle AFTER feat_ts and rebuild: features must be identical
    for res in (300, 3600):
        s = md.series[res]
        mask = s.close_ts > ft
        s.c[mask] *= 7.0; s.h[mask] *= 7.0; s.l[mask] *= 7.0
        s._vol_roll = None
    f2 = features.build_features(md, ev, "pre", ft, cfg.raw["labels"])
    assert f1 is not None and f1 == f2


def test_labels_and_threshold():
    assert features.label_from_return(0.01, 0.005) == "UP"
    assert features.label_from_return(-0.01, 0.005) == "DOWN"
    assert features.label_from_return(0.004, 0.005) == "SIDEWAYS"


def test_sample_threshold_floor_is_cost(small_world):
    cfg, store, evs = small_world
    md = MarketData.from_store(store, "XAUUSD")
    s = features.make_sample(md, evs[45], "XAUUSD", "pre", 3600 * 4, cfg)
    assert s is not None and s.thr >= cfg.all_assets()["XAUUSD"]["cost_bps"] / 1e4 and s.label in ("UP", "DOWN", "SIDEWAYS")
    assert s.end_ts == evs[45].t0 + 4 * 3600


def test_missing_data_gives_none_not_fabrication(small_world):
    cfg, store, evs = small_world
    md = MarketData({})      # no data at all
    assert features.make_sample(md, evs[10], "XAUUSD", "pre", 3600, cfg) is None


# ---------------------------------------------------------------- event study
def S(i, label, cat="CPI", end=None, asset="XAUUSD", z24=0.0):
    return features.Sample(f"e{i}", asset, "pre", 3600, 0, i * 100, end if end is not None else i * 100 + 3600, cat,
                           {"z24": z24, "vol_pct": 0.5, "fcst_vs_prev": 0.0}, 0.001, 0.0, label)


def test_study_insufficient_and_ok_and_leak_filter():
    hist = [S(i, "UP" if i % 2 else "DOWN") for i in range(10)]
    r = eventstudy.study({"z24": 0}, "CPI", "XAUUSD", "pre", 3600, 10**9, hist, min_samples=20)
    assert r["status"] == "INSUFFICIENT HISTORICAL DATA" and r["n"] == 0
    hist = [S(i, "UP" if i % 3 else "DOWN") for i in range(60)]
    r = eventstudy.study({"z24": 0}, "CPI", "XAUUSD", "pre", 3600, 10**9, hist, min_samples=20, k=40)
    assert r["status"] == "OK" and r["n"] == 40 and abs(sum(r["freq"].values()) - 1) < 1e-9
    assert all(r["ci"][l][0] <= r["freq"][l] <= r["ci"][l][1] for l in r["freq"])
    # samples whose outcome ends after the target's feature time must be excluded
    r2 = eventstudy.study({"z24": 0}, "CPI", "XAUUSD", "pre", 3600, 30 * 100, hist, min_samples=5)
    assert r2["n_pool"] == sum(1 for s in hist if s.end_ts <= 3000)
    # other category / other asset never mixed in
    assert eventstudy.study({"z24": 0}, "NFP", "XAUUSD", "pre", 3600, 10**9, hist, 5)["status"] != "OK"


# ---------------------------------------------------------------- ML
def test_ml_insufficient_data_is_reported(base_cfg):
    rep = ml.train([S(i, "UP") for i in range(20)], "pre", 3600, ["XAUUSD"], base_cfg, "v")
    assert rep["status"] == "NOT_TRAINED" and "INSUFFICIENT" in rep["reason"]


def test_walk_forward_has_no_overlap_leak():
    samples = [features.Sample(f"e{i}", "A", "pre", 3600, 0, i * 1000, i * 1000 + 3600, "CPI", {"z24": float(i % 5), "vol_pct": .5}, .1, 0, ["UP", "DOWN", "SIDEWAYS"][i % 3]) for i in range(300)]
    seen = []
    class Spy:
        def __call__(self):
            return self
        def fit(self, X, y):
            seen.append(len(X)); self.classes_ = np.array(sorted(set(y))); return self
        def predict_proba(self, X):
            return np.tile(1 / len(self.classes_), (len(X), len(self.classes_)))
    out = ml.walk_forward_oof(samples, ["z24", "vol_pct"], Spy(), 4)
    assert out is not None and seen == sorted(seen) and len(out[0]) == len(out[1])


def test_calibrator_outputs_valid_distribution():
    rng = np.random.default_rng(0)
    P = rng.dirichlet([1, 1, 1], 400); y = rng.integers(0, 3, 400)
    c = ml.Calibrator().fit(P, y).transform(P)
    assert np.allclose(c.sum(1), 1) and (c >= 0).all()


def test_pure_noise_is_not_validated(small_world, base_cfg):
    cfg, store, evs = small_world
    rng = np.random.default_rng(3)
    names = sorted(cfg.all_assets())
    smp = []
    for i in range(500):
        f = {c: float(rng.normal()) for c in features.feature_columns("pre", names)}
        smp.append(features.Sample(f"e{i}", "XAUUSD", "pre", 3600, i * 5000, i * 5000, i * 5000 + 3600, "CPI", f, .001, 0, ["UP", "DOWN", "SIDEWAYS"][int(rng.integers(3))]))
    rep = ml.train(smp, "pre", 3600, names, cfg, "t", None)
    assert rep["status"] in ("EXPERIMENTAL", "NOT_TRAINED") and rep["status"] != "VALIDATED"


# ---------------------------------------------------------------- risk
RC = {"vol_pct_high": .9, "min_top_probability": .45, "disagreement_min_prob": .4, "imminent_event_minutes": 90}


def ri(**k):
    d = dict(stage="post", event_importance=3, status_label="VALIDATED", probs={"UP": .6, "DOWN": .2, "SIDEWAYS": .2}, vol_pct=.4, staleness_s=100, max_staleness_s=7200)
    d.update(k); return risk.RiskInput(**d)


def test_risk_clean_case_is_acceptable_but_not_signal():
    lvl, dec, why = risk.assess(ri(), RC)
    assert dec.startswith("CONDITIONS ACCEPTABLE") and "not a trade signal" in dec and lvl == "LOW" and not why


@pytest.mark.parametrize("kw,frag", [
    (dict(status_label="EXPERIMENTAL"), "not validated"), (dict(status_label="INSUFFICIENT DATA", probs={}), "insufficient"),
    (dict(staleness_s=99999), "stale"), (dict(vol_pct=.95), "extreme volatility"), (dict(data_missing=True), "missing"),
    (dict(probs={"UP": .34, "DOWN": .33, "SIDEWAYS": .33}), "low confidence"), (dict(deteriorated=True), "deteriorated"),
    (dict(ml_argmax="UP", ml_top_prob=.6, study_argmax="DOWN", study_top_prob=.5), "conflicting"), (dict(late_fraction=.6), "delayed"),
])
def test_risk_hard_stops(kw, frag):
    _, dec, why = risk.assess(ri(**kw), RC)
    assert dec == "STAY AWAY" and any(frag in w for w in why)


def test_pre_event_waits_for_release():
    lvl, dec, _ = risk.assess(ri(stage="pre"), RC)
    assert dec == "WAIT FOR RELEASE" and lvl == "HIGH"


# ---------------------------------------------------------------- storage
def test_event_upsert_never_erases_actual_and_prediction_insert_is_idempotent():
    st = Store(":memory:")
    e = Event("a", "CPI", "USD", 1000, 3, "CPI", "0.3%", "0.1%", "", 0.3, 0.1, None, "s", 1, 1)
    assert st.upsert_event(e) == "new"
    e2 = Event("a", "CPI", "USD", 1000, 3, "CPI", "0.3%", "0.1%", "0.4%", 0.3, 0.1, 0.4, "s", 1, 2, 2)
    assert st.upsert_event(e2) == "updated" and st.get_event("a").actual == 0.4
    e3 = Event("a", "CPI", "USD", 1000, 3, "CPI", "", "", "", None, None, None, "s", 1, 3)       # later blank feed row
    assert st.upsert_event(e3) == "same"
    g = st.get_event("a"); assert g.actual == 0.4 and g.forecast == 0.3
    p = Prediction("p1", "a", "XAUUSD", "pre", 3600, 1, 1000, {}, {}, "UNCERTAIN", "INSUFFICIENT DATA", "none", 0, {}, "HIGH", "STAY AWAY", [], {}, None, None)
    assert st.insert_prediction(p) is True and st.insert_prediction(p) is False
    assert len(st.pending_predictions()) == 1


# ---------------------------------------------------------------- evaluation
def test_pending_not_scored_and_scoring_uses_frozen_threshold(small_world):
    cfg, store, evs = small_world
    ev = evs[50]
    mk = lambda pid, anchor: Prediction(pid, ev.event_id, "XAUUSD", "pre", 3600 * 4, 1, anchor, {"UP": .5, "DOWN": .3, "SIDEWAYS": .2},
                                        {"UP": .33, "DOWN": .33, "SIDEWAYS": .34}, "UP", "VALIDATED", "v", 30, {}, "LOW", "OK", [], {}, 0.0005, 0.5, "CPI")
    store.insert_prediction(mk("old", ev.t0)); store.insert_prediction(mk("future", NOW + 3600))
    r = evaluate.score_pending(store, cfg, NOW)
    assert r["evaluated"] == 1 and r["pending"] == 1
    rows = {x["pred_id"]: x for x in store.db.execute("SELECT * FROM predictions")}
    assert rows["old"]["eval_status"] == "EVALUATED" and rows["old"]["outcome_label"] in ("UP", "DOWN", "SIDEWAYS")
    assert rows["future"]["eval_status"] == "PENDING" and rows["future"]["outcome_label"] is None
    m = evaluate.compute_metrics(store, cfg)
    assert m["overall"]["n"] == 1 and m["pending"] == 1 and "brier" in m["overall"]
    assert "Sample is small" in evaluate.format_summary(m)


def test_summary_with_no_data_claims_nothing(small_world):
    cfg, store, _ = small_world
    t = evaluate.format_summary(evaluate.compute_metrics(store, cfg))
    assert "no accuracy is claimed" in t


# ---------------------------------------------------------------- alerts
class FakeResp:
    def __init__(self, code, body=None):
        self.status_code, self._b = code, body or {}
    def json(self): return self._b


def test_telegram_retry_after_and_failure_hides_token():
    sleeps, calls = [], []
    seq = [FakeResp(429, {"parameters": {"retry_after": 4}}), FakeResp(200)]
    s = alerts.TelegramSender("SECRET:TOKEN", "1", sleep=sleeps.append, post=lambda *a, **k: (calls.append(a[0]), seq.pop(0))[1])
    s.send("hi")
    assert 4.0 in sleeps and len(calls) == 2
    bad = alerts.TelegramSender("SECRET:TOKEN", "1", post=lambda *a, **k: FakeResp(401), sleep=lambda x: None)
    with pytest.raises(alerts.TelegramError) as ei:
        bad.send("x")
    assert "SECRET" not in str(ei.value)
    import requests
    def boom(*a, **k): raise requests.ConnectionError("https://api.telegram.org/botSECRET:TOKEN/sendMessage failed")
    with pytest.raises(alerts.TelegramError) as ei:
        alerts.TelegramSender("SECRET:TOKEN", "1", post=boom, sleep=lambda x: None, retries=1).send("x")
    assert "SECRET" not in str(ei.value)


def test_split_message():
    parts = alerts.split_message("\n".join(["x" * 100] * 50), 1000)
    assert all(len(p) <= 1000 for p in parts) and sum(p.count("x") for p in parts) == 5000


def test_dispatch_dedup_cooldown_and_failure_does_not_record(small_world):
    cfg, store, _ = small_world
    sent = []
    class Snd:
        configured = True
        def __init__(self, fail=False): self.fail = fail
        def send(self, t, m=3500):
            if self.fail: raise alerts.TelegramError("x")
            sent.append(t)
    out = lambda *a, **k: None
    with pytest.raises(alerts.TelegramError):
        alerts.dispatch(store, cfg, Snd(True), "ev", "pre", "t", ["a"], 1000, out=out)
    assert store.alert_last("ev", "pre") is None          # failed send -> retried next run
    assert alerts.dispatch(store, cfg, Snd(), "ev", "pre", "t", ["a"], 1000, out=out) == "sent"
    assert alerts.dispatch(store, cfg, Snd(), "ev", "pre", "t", ["a"], 1100, out=out) == "duplicate"
    assert alerts.dispatch(store, cfg, Snd(), "ev", "pre", "t2", ["b"], 1200, out=out) == "cooldown"
    assert alerts.dispatch(store, cfg, Snd(), "ev", "pre", "t2", ["b"], 1000 + 10**6, out=out) == "sent"
    assert len(sent) == 2


def test_pair_table_lists_every_pair_or_states_disabled(base_cfg):
    pairs = [f"AA{chr(65 + i)}USD"[:6] for i in range(28)]
    lines = alerts.pair_table(base_cfg, {}, pairs, "msg", pairs[:3])
    body = "\n".join(lines)
    assert all(p in body for p in pairs)
    assert alerts.pair_table(base_cfg, {}, [], "FOREX DISABLED: no list", [])[0].startswith("FOREX: FOREX DISABLED")


# ---------------------------------------------------------------- validation gate discriminates signal from noise (synthetic)
def _synthetic_samples(strength, seed, cols, n=600):
    rng = np.random.default_rng(seed); out = []
    for i in range(n):
        f = {c: 0.0 for c in cols}; f["z24"] = float(rng.normal())
        s = strength * f["z24"] + rng.normal()
        lab = "UP" if s > 0.5 else ("DOWN" if s < -0.5 else "SIDEWAYS")
        t = i * 5000
        out.append(features.Sample(f"e{i}", "XAUUSD", "pre", 3600, t, t, t + 3600, "CPI", f, .001, 0, lab))
    return out


def test_validation_gate_accepts_real_signal_and_rejects_noise(base_cfg, tmp_path):
    cols = features.feature_columns("pre", ["XAUUSD"])
    good = ml.train(_synthetic_samples(2.0, 1, cols), "pre", 3600, ["XAUUSD"], base_cfg, "t", str(tmp_path))
    assert good["status"] == "VALIDATED" and good["test"]["brier_skill"] > 0.2 and good["test"]["n"] >= 40
    art = ml.load_artifact(str(tmp_path), "pre", 3600)
    p = ml.predict_proba(art, {**{c: 0.0 for c in cols}, "z24": 2.0})
    assert abs(p.sum() - 1) < 1e-9 and p[0] > p[1]          # strongly positive z24 -> UP more likely than DOWN
    bad = ml.train(_synthetic_samples(0.0, 1, cols), "pre", 3600, ["XAUUSD"], base_cfg, "t", None)
    assert bad["status"] == "EXPERIMENTAL"
