"""Explainable directional scoring. score = 100 * sum(weight*evidence) / sum(applicable nominal weight): missing-but-applicable
evidence shrinks the score toward neutral. Price evidence may confirm/contradict but can never create a preference by itself.
Overrides (stale/missing data, closed market, event risk, pending actuals, extreme volatility, conflicts) can force WAIT / STAY AWAY."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .. import eventstudy
from ..classify import hawkish_rule, surprise_sign
from ..features import hourly_vol, relevant_assets_cfg, vol_percentile
from ..prices import MarketData
from ..symbols import asset_kind
from . import evidence as ev_
from .models import (AWAY, HZ_NAME, LONG, MODEL, PRICE, SHORT, WAIT, CurrencyOutlook, Evidence, EventCtx, Recommendation)
from .validation import validation_status

LEVELS = ["LOW", "MODERATE", "HIGH"]


def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ------------------------------------------------------------------------------------------------ pure scoring core
def combine(evs: List[Evidence], full_coverage: float = 0.7) -> Dict[str, float]:
    """Weighted average direction of the evidence that EXISTS (-100..+100), shrunk toward 0 when coverage of the applicable
    evidence weight is below `full_coverage`. So a single strong driver cannot masquerade as broad confirmation."""
    app = [e for e in evs if e.applicable]
    W = sum(e.weight for e in app)
    avail = [e for e in app if e.value is not None]
    Wa = sum(e.weight for e in avail)
    if W <= 0 or Wa <= 0:
        return {"score": 0.0, "coverage": 0.0, "agreement": 0.0, "conflict": 0.0, "fund": 0.0}
    coverage = Wa / W
    shrink = min(1.0, coverage / max(full_coverage, 1e-6))
    score = 100.0 * shrink * sum(e.weight * e.value for e in avail) / Wa
    fund = 100.0 * shrink * sum(e.weight * e.value for e in avail if e.kind != PRICE) / Wa
    sign = 1 if score > 0 else -1 if score < 0 else 0
    pro = sum(e.weight * abs(e.value) for e in avail if e.value * sign > 0)
    con = sum(e.weight * abs(e.value) for e in avail if e.value * sign < 0)
    tot = pro + con
    return {"score": score, "coverage": coverage, "fund": fund,
            "agreement": (pro / tot) if (tot > 0 and sign) else 0.0, "conflict": (con / tot) if (tot > 0 and sign) else 0.0}


@dataclass
class Flags:
    freshness: str = "RECENT"
    age_s: Optional[int] = None
    short: bool = False
    blackout_pre: Optional[str] = None
    blackout_post: Optional[str] = None
    pending_post: Optional[str] = None
    pre_event: Optional[str] = None
    pre_importance: int = 0
    two_sided_wait: bool = True
    extreme_vol: bool = False
    vol_pct: Optional[float] = None
    validated: bool = False
    delayed: bool = False


def decide(evs: List[Evidence], fl: Flags, th: dict) -> Tuple[str, float, str, str, List[str], List[str], List[str], Dict[str, float]]:
    """Returns (direction, score, confidence, status, supporting, contradicting, risks, stats)."""
    st = combine(evs, float(th.get("full_coverage", 0.7)))
    score = st["score"]
    sign = 1 if score > 0 else -1 if score < 0 else 0
    ranked = sorted([e for e in evs if e.value is not None and e.applicable], key=lambda e: -abs(e.weight * e.value))
    sup = [e.text for e in ranked if e.value * sign > 0][:4]
    con = [e.text for e in ranked if e.value * sign < 0][:3]
    risks: List[str] = []

    def out(direction, status, why, conf="n/a"):
        return direction, score, conf, status, sup, con, [why] + risks, st

    if fl.freshness == "UNAVAILABLE":
        return out(AWAY, "UNAVAILABLE", "market data unavailable (cannot assess)")
    if fl.freshness == "CLOSED":
        return out(WAIT, "MARKET CLOSED", "market closed (weekly closure); outlook resumes at the open")
    if fl.freshness == "STALE":
        return out(AWAY, "STALE", f"price data stale ({(fl.age_s or 0) // 60} min old)")
    if fl.blackout_pre and fl.short:
        return out(AWAY, "EVENT RISK", f"{fl.blackout_pre} imminent (spike/whipsaw risk)")
    if fl.blackout_post and fl.short:
        return out(AWAY, "EVENT RISK", f"{fl.blackout_post}: post-release volatility spike window")
    if fl.pending_post:
        return out(WAIT, "PENDING", f"post-news recommendation pending: actual for {fl.pending_post} not available yet")
    if fl.pre_event and fl.short and fl.two_sided_wait and fl.pre_importance >= 3:
        return out(WAIT, "EVENT RISK", f"{fl.pre_event} pending: outcome is two-sided, no bias for short horizons")
    if fl.extreme_vol:
        return out(AWAY, "EXTREME VOLATILITY", f"extreme volatility (hourly vol at {fl.vol_pct:.0%} percentile)")
    if st["coverage"] < float(th["min_coverage"]):
        return out(WAIT, "INSUFFICIENT EVIDENCE", f"only {st['coverage']:.0%} of applicable evidence has data")
    need = float(th["direction_score"]) * (float(th["pre_news_score_multiplier"]) if fl.pre_event else 1.0)
    if abs(score) < need:
        return out(WAIT, "NO CLEAR PREFERENCE", f"score {score:+.0f} inside neutral band (+/-{need:.0f})")
    if abs(st["fund"]) < float(th["fundamental_min"]) or st["fund"] * score <= 0:
        return out(WAIT, "NO FUNDAMENTAL DRIVER", "price momentum alone does not create a preference")
    if st["conflict"] > float(th["max_conflict"]):
        return out(WAIT, "CONFLICTING", f"conflicting evidence ({st['conflict']:.0%} of weight opposes)")
    conf = "LOW"
    if abs(score) >= need and st["agreement"] >= 0.65 and st["coverage"] >= 0.5:
        conf = "MODERATE"
    if abs(score) >= float(th["strong_score"]) and st["agreement"] >= 0.8 and st["coverage"] >= 0.7 and fl.validated and fl.freshness == "RECENT":
        conf = "HIGH"
    if fl.pre_event:
        conf = min(conf, "MODERATE", key=LEVELS.index)
        risks.append(f"PRE-NEWS BIAS for {fl.pre_event}: not a confirmed prediction")
    if fl.pre_event and fl.short:
        conf = "LOW"
    if fl.freshness == "DELAYED":
        conf = "LOW"; risks.append("price data delayed")
    if not fl.validated:
        conf = min(conf, "MODERATE", key=LEVELS.index)
        risks.append("recommendation logic UNVALIDATED (default thresholds; see live track record)")
    return (LONG if score > 0 else SHORT), score, conf, ("VALIDATED" if fl.validated else "UNVALIDATED"), sup, con, risks, st


# ------------------------------------------------------------------------------------------------ engine
class Engine:
    def __init__(self, cfg, store, now_ts: int, mds: Dict[str, MarketData], hist_post: Optional[dict] = None,
                 deteriorated: bool = False, crypto_ctx=None):
        self.cfg, self.store, self.now, self.mds, self.hist_post, self.det, self.cx = cfg, store, now_ts, mds, hist_post, deteriorated, crypto_ctx
        self.rc = cfg.raw["recommend"]
        self.assets = cfg.all_assets()
        self.horizons = [h * 60 for h in cfg.horizons]
        self.short = {int(h) * 60 for h in self.rc["short_horizons_minutes"]}
        self.macro = ev_.macro_scores(store, cfg, now_ts)
        self.tone = ev_.tone_scores(store, cfg, now_ts)
        self.rates = ev_.rates_score(store, cfg, now_ts)
        self.risk = ev_.risk_sentiment(cfg, mds, now_ts)
        self.valid = {h: validation_status(store, cfg, h)["validated"] for h in self.horizons}
        self.events = self._event_ctx()
        self.pair_z: Dict[int, Dict[str, float]] = {h: {} for h in self.horizons}
        self.pair_ts: Dict[Tuple[str, int], int] = {}
        for a, m in self.assets.items():
            if asset_kind(m) == "forex":
                for h in self.horizons:
                    r = ev_.price_z(mds.get(a), now_ts, h, cfg.raw["labels"])
                    if r:
                        self.pair_z[h][a] = r[0]; self.pair_ts[(a, h)] = r[1]
        self.strength = {h: ev_.currency_strength(z) for h, z in self.pair_z.items()}

    # ---- events around now (high-impact only) ------------------------------------------------------------------
    def _event_ctx(self) -> List[EventCtx]:
        rc = self.rc
        mn = int(rc["alert_min_importance"])
        t_from, t_to = self.now - int(rc["post_window_minutes"]) * 60, self.now + max(int(m) for m in rc["pre_checks_minutes"]) * 60
        out = []
        for e in self.store.events_between(t_from, t_to + 1, mn):
            aff = relevant_assets_cfg(self.cfg, e.currency)
            if e.t0 > self.now:
                out.append(EventCtx(e.event_id, e.name, e.currency, e.t0, e.importance, "pre", (e.t0 - self.now) / 60, e.forecast_raw, e.previous_raw, "", aff))
            else:
                ph = "post" if e.actual is not None else "post_pending"
                out.append(EventCtx(e.event_id, e.name, e.currency, e.t0, e.importance, ph, (self.now - e.t0) / 60, e.forecast_raw, e.previous_raw, e.actual_raw, aff))
        return out

    # ---- evidence ----------------------------------------------------------------------------------------------
    def _w(self, kind: str, h: int) -> dict:
        key = "forex" if kind == "forex" else "crypto" if kind == "crypto" else "other"
        return self.rc["weights"][key][h // 60]

    def _model(self, asset: str, h: int, w: float) -> Evidence:
        r = self.store.db.execute("SELECT * FROM predictions WHERE asset=? AND horizon_s=? AND status_label!='INSUFFICIENT DATA' AND anchor_ts+horizon_s>? "
                                  "ORDER BY created_ts DESC LIMIT 1", (asset, h, self.now)).fetchone()
        if r is None:
            return Evidence("model", "model/history", None, w, "no active model/history forecast for this window", MODEL)
        p = json.loads(r["probs"])
        if not p:
            return Evidence("model", "model/history", None, w, "forecast had no probabilities", MODEL)
        valid = r["status_label"] == "VALIDATED"
        if valid and self.det:
            return Evidence("model", "model/history", None, w, "validated model excluded: live performance deteriorated vs baseline", MODEL)
        f = 1.0 if valid else 0.25
        return Evidence("model", "model/history", clip(p["UP"] - p["DOWN"]), w * f,
                        f"{'validated ML' if valid else 'EXPERIMENTAL model/history (down-weighted x0.25)'}: P(up) {p['UP']:.0%} vs P(down) {p['DOWN']:.0%}", MODEL, True, r["created_ts"])

    def _reaction(self, asset: str, h: int, w: float) -> Evidence:
        label = "historical reaction"
        if self.hist_post is None:
            return Evidence("reaction", label, None, w, "historical reaction analysis not run this cycle", MODEL)
        best = None
        for e in self.store.events_between(self.now - int(self.rc["lookback_hours"]) * 3600, self.now + 1, int(self.rc["min_event_importance"])):
            if e.actual is None or e.forecast is None or asset not in relevant_assets_cfg(self.cfg, e.currency):
                continue
            sg = surprise_sign(e.actual, e.forecast)
            if not sg or hawkish_rule(e.category, e.name) is None:
                continue
            d = ev_.decay(self.now - e.t0, float(self.rc["halflife_hours"]))
            if best is None or d * e.importance > best[0]:
                best = (d * e.importance, e, sg, d)
        if best is None:
            return Evidence("reaction", label, None, w, "no recent released event with a measurable surprise", MODEL)
        _, e, sg, d = best
        st = eventstudy.scenario_stats(self.hist_post.get(("post", h), []), asset, e.category, h, sg, self.now,
                                       int(self.cfg.g("study", "min_samples", default=20)), exclude_event_id=e.event_id)
        if st["status"] != "OK":
            return Evidence("reaction", label, None, w, f"{e.category} reactions: INSUFFICIENT HISTORICAL DATA (n={st['n']})", MODEL)
        f = st["freq"]
        return Evidence("reaction", label, clip((f["UP"] - f["DOWN"]) * d), w,
                        f"after comparable {'ABOVE' if sg > 0 else 'BELOW'}-forecast {e.category}: UP {f['UP']:.0%} / DOWN {f['DOWN']:.0%} (n={st['n']})", MODEL, True, e.t0)

    def _macro_text(self, ccy_list: List[Tuple[str, int]]) -> str:
        bits = []
        for c, sgn in ccy_list:
            it = self.macro[c]["items"]
            if it:
                bits.append(it[0][1])
        return "; ".join(bits[:2]) if bits else "no recent released surprises for the currencies involved"

    def evidence_for(self, name: str, meta: dict, kind: str, h: int) -> List[Evidence]:
        w = self._w(kind, h)
        evs: List[Evidence] = []
        base, quote = meta.get("base"), meta.get("quote")
        usd_m = self.macro["USD"]["score"]
        # ---- macro / tone / rates / risk --------------------------------------------------------------------
        if kind == "forex":
            mb, mq = self.macro.get(base, {}).get("score"), self.macro.get(quote, {}).get("score")
            val = None if (mb is None and mq is None) else clip(((mb or 0.0) - (mq or 0.0)) * 0.8)
            evs.append(Evidence("macro", "macro surprises", val, w["macro"], self._macro_text([(base, 1), (quote, -1)]) if val is not None else "no recent released surprises for these currencies"))
            tb, tq = self.tone.get(base, {}).get("score"), self.tone.get(quote, {}).get("score")
            tv = None if (tb is None and tq is None) else clip(((tb or 0.0) - (tq or 0.0)))
            evs.append(Evidence("tone", "central-bank tone", tv, w["tone"], "official central-bank headline tone (keyword, coarse)" if tv is not None else "no tone-bearing official headlines"))
            usd_involved = "USD" in (base, quote)
            rv = None if (self.rates is None or not usd_involved) else clip(self.rates["score"] * (1 if base == "USD" else -1))
            evs.append(Evidence("rates", "US yields", rv, w["rates"], self.rates["text"] if (self.rates and usd_involved) else "US yield data unavailable",
                                applicable=usd_involved, data_ts=self.rates["ts"] if self.rates else None))
            beta = float(self.rc["risk_beta"].get(base, 0.0)) - float(self.rc["risk_beta"].get(quote, 0.0))
            evs.append(Evidence("risk", "risk sentiment", None if self.risk is None else clip(self.risk["score"] * beta / 2), w["risk"],
                                f"{self.risk['text']} x risk-beta {beta:+.1f}" if self.risk else "risk-sentiment proxy unavailable", applicable=abs(beta) > 1e-9))
        elif kind == "crypto":
            evs.append(Evidence("macro", "macro surprises (USD)", None if usd_m is None else clip(-usd_m * 0.6), w["macro"],
                                ("textbook inverse for risk assets: " + self._macro_text([("USD", 1)])) if usd_m is not None else "no recent USD surprises"))
            evs.append(Evidence("rates", "US yields", None if self.rates is None else clip(-self.rates["score"] * 0.6), w["rates"], self.rates["text"] if self.rates else "US yield data unavailable"))
            beta = float(self.rc["risk_beta"].get(base, 1.0))
            evs.append(Evidence("risk", "risk sentiment", None if self.risk is None else clip(self.risk["score"] * beta / 1.2), w["risk"],
                                self.risk["text"] if self.risk else "risk-sentiment proxy unavailable"))
            mv, mt, mts = self.cx.market(h) if self.cx else (None, "crypto context unavailable", None)
            adj, at = self.cx.alt_adjust(base) if self.cx else (0.0, "")
            evs.append(Evidence("cryptomkt", "crypto market", None if mv is None else clip(mv + adj), w["cryptomkt"], mt + (f"; {at}" if at else ""), data_ts=mts))
        else:   # metal / index / dollar_index
            sgn = {"dollar_index": 1.0, "metal": -1.0, "index": -0.5}[kind]
            evs.append(Evidence("macro", "macro surprises (USD)", None if usd_m is None else clip(usd_m * sgn), w["macro"],
                                (self._macro_text([("USD", 1)]) + f" (textbook sign {sgn:+.1f} for {kind})") if usd_m is not None else "no recent USD surprises"))
            evs.append(Evidence("rates", "US yields", None if self.rates is None else clip(self.rates["score"] * sgn), w["rates"], self.rates["text"] if self.rates else "US yield data unavailable"))
            beta = float(self.rc["risk_beta"].get(base, 0.0))
            evs.append(Evidence("risk", "risk sentiment", None if self.risk is None else clip(self.risk["score"] * beta), w["risk"],
                                self.risk["text"] if self.risk else "risk-sentiment proxy unavailable", applicable=abs(beta) > 1e-9))
        # ---- model / history -----------------------------------------------------------------------------------
        evs.append(self._model(name, h, w["model"]))
        evs.append(self._reaction(name, h, w["reaction"]))
        # ---- price confirmation --------------------------------------------------------------------------------
        pz = self.pair_z[h].get(name) if kind == "forex" else None
        r = None if pz is not None else ev_.price_z(self.mds.get(name), self.now, h, self.cfg.raw["labels"])
        z = pz if pz is not None else (r[0] if r else None)
        evs.append(Evidence("price", "price trend", None if z is None else math.tanh(z / 2), w["price"],
                            f"{name} trend {z:+.1f} sigma over {ev_.LOOKBACK[h][0] // 3600}h" if z is not None else "price trend unavailable", PRICE))
        if kind == "forex":
            s = self.strength.get(h)
            if s is None or z is None:
                evs.append(Evidence("strength", "currency strength", None, w["strength"], "currency-strength meter unavailable (needs >=12 fresh pairs)", PRICE))
            else:
                d = s[base] - s[quote]
                evs.append(Evidence("strength", "currency strength", math.tanh(d / 2), w["strength"],
                                    f"common currency strength {base} {s[base]:+.1f} vs {quote} {s[quote]:+.1f} (pair-specific residual {z - d:+.1f})", PRICE))
        return evs

    # ---- flags -----------------------------------------------------------------------------------------------------
    def flags_for(self, name: str, meta: dict, kind: str, h: int) -> Flags:
        md = self.mds.get(name)
        fresh, last_ts, age = ev_.freshness(md, self.now, kind, self.cfg)
        fl = Flags(freshness=fresh, age_s=age, short=h in self.short, validated=self.valid.get(h, False),
                   two_sided_wait=bool(self.rc["pre_news_two_sided_wait"]), delayed=fresh == "DELAYED")
        labels = self.cfg.raw["labels"]
        lb, mn = int(labels.get("vol_lookback_hours", 24)), int(labels.get("min_vol_returns", 12))
        vp = vol_percentile(md, self.now, lb, mn) if md else None
        fl.vol_pct = vp
        fl.extreme_vol = vp is not None and vp >= float(self.cfg.raw["risk"]["vol_pct_high"])
        for e in self.events:
            if name not in e.affected:
                continue
            if e.phase == "pre":
                if e.minutes <= float(self.rc["blackout_pre_minutes"]):
                    fl.blackout_pre = f"{e.currency} {e.name} in {int(e.minutes)} min"
                if fl.pre_event is None or e.importance > fl.pre_importance:
                    fl.pre_event, fl.pre_importance = f"{e.currency} {e.name}", e.importance
            elif e.phase == "post":
                if e.minutes < float(self.rc["blackout_post_minutes"]):
                    fl.blackout_post = f"{e.currency} {e.name} released {int(e.minutes)} min ago"
            else:
                if e.minutes >= float(self.rc["blackout_post_minutes"]):
                    fl.pending_post = f"{e.currency} {e.name}"
        return fl

    # ---- recommendations ---------------------------------------------------------------------------------------
    def recommend(self, name: str, h: int) -> Recommendation:
        meta = self.assets[name]
        kind = asset_kind(meta)
        evs = self.evidence_for(name, meta, kind, h)
        fl = self.flags_for(name, meta, kind, h)
        direction, score, conf, status, sup, con, risks, st = decide(evs, fl, self.rc["thresholds"])
        md = self.mds.get(name)
        _, last_ts, _ = ev_.freshness(md, self.now, kind, self.cfg)
        vol = hourly_vol(md, self.now, int(self.cfg.raw["labels"].get("vol_lookback_hours", 24)), int(self.cfg.raw["labels"].get("min_vol_returns", 12))) if md else None
        thr = None
        if vol:
            thr = max(float(self.cfg.raw["labels"]["sideways_k"]) * vol * math.sqrt(h / 3600.0), float(meta.get("cost_bps", 2)) / 1e4)
        stage, event = "SCAN", None
        aff = [e for e in self.events if name in e.affected]
        if fl.pending_post:
            stage, event = "POST-NEWS PENDING", fl.pending_post
        elif any(e.phase == "post" for e in aff):
            e = min((e for e in aff if e.phase == "post"), key=lambda x: x.minutes); stage, event = "POST-NEWS", f"{e.currency} {e.name}"
        elif fl.pre_event:
            stage, event = "PRE-NEWS BIAS", fl.pre_event
        return Recommendation(name, h, direction, round(score, 1), conf, status, sup, con, risks, last_ts, fl.freshness, event, stage, self.now,
                              round(st["coverage"], 3), "", kind, thr, fl.vol_pct, fl.validated,
                              {e.key: (None if e.value is None else round(e.value, 3)) for e in evs if e.applicable})

    def currency_outlooks(self) -> Dict[str, CurrencyOutlook]:
        out = {}
        s1 = self.strength.get(3600) or self.strength.get(14400)
        for c in ev_.CURRENCIES:
            parts = []
            m, t = self.macro[c]["score"], self.tone[c]["score"]
            if m is not None:
                parts.append((0.6, m))
            if t is not None:
                parts.append((0.2, t))
            if c == "USD" and self.rates:
                parts.append((0.2, self.rates["score"]))
            reasons = [x[1] for x in self.macro[c]["items"][:2]] + [x[1] for x in self.tone[c]["items"][:1]]
            if c == "USD" and self.rates:
                reasons.append(self.rates["text"])
            if not parts:
                out[c] = CurrencyOutlook(c, 0.0, "NEUTRAL", "n/a", ["no recent released data / tone for this currency"], s1.get(c) if s1 else None, None)
                continue
            sc = 100.0 * sum(w * v for w, v in parts) / sum(w for w, _ in parts)
            d = "BULLISH" if sc >= 20 else "BEARISH" if sc <= -20 else "NEUTRAL"
            n_items = len(self.macro[c]["items"]) + len(self.tone[c]["items"])
            agree = all(v * sc > 0 for _, v in parts) if sc else False
            conf = "MODERATE" if (n_items >= 2 and agree and abs(sc) >= 30) else "LOW"
            out[c] = CurrencyOutlook(c, round(sc, 1), d, conf, reasons, s1.get(c) if s1 else None, self.macro[c]["last_ts"])
        return out

    def run(self) -> dict:
        recs = {name: {h: self.recommend(name, h) for h in self.horizons} for name in self.assets}
        return {"recs": recs, "currency": self.currency_outlooks(), "events": self.events, "now": self.now,
                "crypto": (self.cx.market(3600)[1] if self.cx else "crypto context unavailable"),
                "validation": {HZ_NAME[h]: self.valid[h] for h in self.horizons}}
