"""One consolidated, grouped Telegram message per analysis cycle (plain text). Common drivers are explained once; pairs are
grouped by their strongest shared currency; identical WAIT groups are compressed. Nothing here invents data: every line comes
from the Recommendation/CurrencyOutlook objects."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..classify import hawkish_rule, theoretical_impact
from ..symbols import asset_kind
from .models import AWAY, CODE, HZ_NAME, LONG, SHORT, WAIT
from .schedule import DueItem

GROUP_ORDER = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]
CONF_SHORT = {"LOW": "LOW", "MODERATE": "MOD", "HIGH": "HIGH", "n/a": ""}
WORD = {LONG: "LONG PREFERENCE", SHORT: "SHORT PREFERENCE", WAIT: "WAIT", AWAY: "STAY AWAY"}


def local_label(ts: Optional[int], tzname: str) -> str:
    if ts is None:
        return "n/a"
    base = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if tzname and tzname.upper() != "UTC":
        try:
            from zoneinfo import ZoneInfo
            dt = datetime.fromtimestamp(int(ts), tz=ZoneInfo(tzname))
            return f"{base} ({dt.strftime('%H:%M')} {dt.tzname()})"
        except Exception:
            return f"{base} (timezone '{tzname}' unavailable)"
    return base


def group_of(pair: str) -> str:
    for c in GROUP_ORDER:
        if c in (pair[:3], pair[3:]):
            return c
    return "OTHER"


def _code(r) -> str:
    return "-" if r.status == "UNAVAILABLE" else CODE[r.direction]


def _line(name: str, by_h: Dict[int, object], horizons: List[int]) -> str:
    r1 = by_h[3600]
    directional = r1.direction in (LONG, SHORT)
    conf = CONF_SHORT.get(r1.confidence, "")
    bracket = "1h" + (f", {conf}" if directional and conf else "")
    others = " · ".join(f"{HZ_NAME[h]} {_code(by_h[h])}" for h in horizons if h != 3600)
    tag = "" if directional or r1.status == "NO CLEAR PREFERENCE" else f" [{r1.status.lower()}]"
    return f"{name} — {WORD[r1.direction]} ({bracket}){tag} | {others}"


def _group_lines(title: str, names: List[str], recs, horizons) -> List[str]:
    keys = {n: tuple(_code(recs[n][h]) for h in horizons) + (recs[n][3600].status,) for n in names}
    non_dir = all(recs[n][3600].direction not in (LONG, SHORT) for n in names)
    if len(names) > 1 and non_dir and len(set(keys.values())) == 1:
        r = recs[names[0]]
        others = " · ".join(f"{HZ_NAME[h]} {_code(r[h])}" for h in horizons if h != 3600)
        return [f"{title}: {', '.join(names)} — all {WORD[r[3600].direction]} (1h) [{r[3600].status.lower()}] | {others}"]
    return [title] + ["  " + _line(n, recs[n], horizons) for n in names]


def scenario_line(cfg, di: DueItem, affected_forex: List[str]) -> Optional[str]:
    ev = di.event
    hk = hawkish_rule(ev.category, ev.name)
    if hk is None or not affected_forex:
        return None
    meta = cfg.all_assets()
    up, dn = [], []
    for p in affected_forex:
        th, _ = theoretical_impact(ev.currency, hk, 1, meta[p])
        (up if th == "BULLISH" else dn if th == "BEARISH" else []).append(p)
    if not (up or dn):
        return None
    cut = lambda L: ", ".join(L[:6]) + ("..." if len(L) > 6 else "")
    return (f"SCENARIOS (textbook theory, not a forecast): if {ev.currency} {ev.name} comes ABOVE forecast -> "
            f"up-leaning: {cut(up) or 'none'}; down-leaning: {cut(dn) or 'none'}. BELOW forecast -> reverse. Actual reaction may differ.")


def material(result: dict, due: List[DueItem]) -> List[str]:
    """Content digest used for de-duplication: only coarse direction classes at 1h/4h (WAIT and STAY AWAY merged) + currency outlooks,
    so score drift, timestamps and status flapping never trigger a new message."""
    norm = lambda d: d if d in (LONG, SHORT) else "NEUTRAL"
    m = [f"{n}:{norm(r[3600].direction)}:{norm(r[14400].direction)}" for n, r in sorted(result["recs"].items())]
    m += [f"{c}:{o.direction}" for c, o in sorted(result["currency"].items())]
    m += sorted(d.event.event_id + ":" + d.key for d in due)
    return m


def build_report(cfg, result: dict, due: List[DueItem], now_ts: int, scan_changes: Optional[List[str]] = None) -> str:
    rc = cfg.raw["recommend"]
    tz = rc.get("display_timezone", "UTC")
    recs, cur, horizons = result["recs"], result["currency"], sorted({h for r in result["recs"].values() for h in r})
    assets = cfg.all_assets()
    forex = [n for n in assets if asset_kind(assets[n]) == "forex"]
    crypto = [n for n in assets if asset_kind(assets[n]) == "crypto"]
    other = [n for n in assets if asset_kind(assets[n]) in ("metal", "index", "dollar_index")]
    L: List[str] = ["UAMIS | MARKET DIRECTION REPORT"]
    kinds = {d.kind for d in due}
    parts = []
    if "pre" in kinds:
        parts.append("PRE-NEWS BIAS (not a confirmed prediction)")
    if "post" in kinds:
        parts.append("POST-NEWS")
    if "pending" in kinds:
        parts.append("POST-NEWS PENDING (actual not yet available)")
    L.append("Analysis: " + (" + ".join(parts) if parts else ("SCHEDULED SCAN (meaningful directional change)" if scan_changes else "MARKET SNAPSHOT (no scheduled release due)")))
    for d in due:
        e = d.event
        extra = (f"Forecast {e.forecast_raw or 'n/a'} | Previous {e.previous_raw or 'n/a'}" + (f" | Actual {e.actual_raw}" if e.actual_raw else " | Actual: not released"))
        L.append(f"Event: {e.currency} {e.name} — {local_label(e.t0, tz)}")
        L.append(f"  {extra}")
    fr = {}
    for n in forex + crypto + other:
        f = recs[n][3600].freshness
        fr[f] = fr.get(f, 0) + 1
    L.append(f"Updated: {local_label(now_ts, tz)} | Data: " + ", ".join(f"{v} {k}" for k, v in sorted(fr.items())))
    val = [h for h, v in result["validation"].items() if v]
    L.append("Status: " + ("recommendation logic VALIDATED on live outcomes for " + ", ".join(val) if val else
                           "UNVALIDATED — default thresholds, not yet proven by live outcomes. Use as a cautious filter only."))
    if scan_changes:
        L.append("Changes: " + "; ".join(scan_changes[:6]))
    L.append("")
    # ---- currency outlook ----------------------------------------------------------------------------------
    L.append("CURRENCY OUTLOOK (macro surprises, tone, USD yields)")
    quiet = []
    for c in GROUP_ORDER:
        o = cur[c]
        if o.direction == "NEUTRAL" and not o.reasons[0].startswith(("no recent",)) and abs(o.score) < 5:
            quiet.append(c); continue
        if o.direction == "NEUTRAL" and o.reasons[0].startswith("no recent"):
            quiet.append(c); continue
        L.append(f"{c}: {o.direction} ({o.score:+.0f}, {o.confidence}) — {o.reasons[0][:120]}")
    if quiet:
        L.append("Neutral / no recent data: " + ", ".join(quiet))
    usd = cur["USD"]
    if usd.direction != "NEUTRAL" and forex:
        bull, bear = [], []
        for n in forex:
            if "USD" not in (n[:3], n[3:]):
                continue
            d = recs[n][3600].direction
            if d not in (LONG, SHORT):
                continue
            usd_long = (n[:3] == "USD") == (d == LONG)
            (bull if usd_long else bear).append(f"{'LONG' if d == LONG else 'SHORT'} {n}")
        L.append("USD-bullish expressions (1h): " + (", ".join(bull) or "none") + " | USD-bearish: " + (", ".join(bear) or "none"))
    for d in due:
        if d.kind == "pre":
            aff = [n for n in forex if d.event.currency in (n[:3], n[3:])]
            sl = scenario_line(cfg, d, aff)
            if sl:
                L.append(sl)
    L.append("")
    # ---- forex -----------------------------------------------------------------------------------------------
    if forex:
        L.append(f"FOREX — {len(forex)} pairs  (code per horizon: L=LONG PREF, S=SHORT PREF, W=WAIT, X=STAY AWAY, -=unavailable)")
        for g in GROUP_ORDER:
            names = [n for n in forex if group_of(n) == g]
            if names:
                L += _group_lines(f"{g} {'PAIRS' if g == 'USD' else 'CROSSES'}", names, recs, horizons)
    else:
        L.append("FOREX: disabled (config/forex_pairs.txt must contain exactly the configured number of valid pairs)")
    if other:
        L.append("")
        L.append("OTHER MARKETS")
        L += ["  " + _line(n, recs[n], horizons) for n in other]
    # ---- crypto -----------------------------------------------------------------------------------------------
    L.append("")
    L.append("CRYPTO OUTLOOK")
    vals = [recs[n][3600].components.get("cryptomkt") for n in crypto]
    vals = [v for v in vals if v is not None]
    if vals:
        mv = sum(vals) / len(vals)
        lab = "LEAN LONG" if mv >= 0.2 else "LEAN SHORT" if mv <= -0.2 else "NEUTRAL"
        L.append(f"Broad crypto market: {lab} ({mv:+.2f}) — {result['crypto']}")
    else:
        L.append("Broad crypto market: UNAVAILABLE — " + result["crypto"])
        mv = None
    order = sorted(crypto, key=lambda n: (n not in ("BTCUSD", "ETHUSD"), n))
    for n in order:
        line = "  " + _line(n, recs[n], horizons)
        pz = recs[n][3600].components.get("price")
        if mv is not None and pz is not None and abs(mv) >= 0.2 and abs(pz) >= 0.2 and mv * pz < 0:
            line += "  ! own price trend diverges from market bias"
        L.append(line)
    L.append("")
    # ---- timeframe summary / overall / risk ------------------------------------------------------------------
    L.append("TIMEFRAME SUMMARY")
    allnames = forex + crypto + other
    for h in horizons:
        codes = [_code(recs[n][h]) for n in allnames]
        if all(c == "-" for c in codes):
            L.append(f"{HZ_NAME[h]}: unavailable"); continue
        cnt = {k: codes.count(k) for k in "LSWX"}
        bull = sum(1 for n in forex if "USD" in (n[:3], n[3:]) and recs[n][h].direction in (LONG, SHORT)
                   and ((n[:3] == "USD") == (recs[n][h].direction == LONG)))
        bear = sum(1 for n in forex if "USD" in (n[:3], n[3:]) and recs[n][h].direction in (LONG, SHORT)) - bull
        usd = ("USD-bullish lean" if bull >= 2 * max(bear, 1) and bull >= 2 else "USD-bearish lean" if bear >= 2 * max(bull, 1) and bear >= 2 else "USD mixed/none")
        L.append(f"{HZ_NAME[h]}: L{cnt['L']} S{cnt['S']} W{cnt['W']} X{cnt['X']} | {usd} ({bull} vs {bear})")
    confl = []
    for n in allnames:
        ds = {h: recs[n][h].direction for h in horizons}
        if LONG in ds.values() and SHORT in ds.values():
            lo = next(h for h in horizons if ds[h] in (LONG, SHORT))
            hi = next(h for h in reversed(horizons) if ds[h] in (LONG, SHORT) and ds[h] != ds[lo])
            confl.append(f"{n} ({HZ_NAME[lo]} {CODE[ds[lo]]} vs {HZ_NAME[hi]} {CODE[ds[hi]]})")
    if confl:
        L.append("Horizon conflicts: " + ", ".join(confl[:5]))
    due_ids = {d.event.event_id for d in due}
    affected = {a for e in result["events"] if e.event_id in due_ids for a in e.affected}
    scope = [n for n in allnames if n in affected] or allnames      # event-relevant instruments when an event triggered the report
    c1 = [recs[n][3600].direction for n in scope]
    nl, ns = c1.count(LONG), c1.count(SHORT)
    overall = "WAIT" if nl + ns < 3 else "LONG BIAS" if nl >= 2 * ns else "SHORT BIAS" if ns >= 2 * nl else "MIXED"
    L.append("")
    L.append(f"OVERALL PREFERENCE (1h, count of {nl} LONG vs {ns} SHORT preferences; offsetting pair constructions can cancel - read the USD line): {overall}")
    # risk
    reasons, level = [], "LOW"
    ev_pre = [e for e in result["events"] if e.phase == "pre"]
    imminent = [e for e in ev_pre if e.minutes <= float(rc["blackout_pre_minutes"])]
    if imminent:
        level = "HIGH"; reasons.append(f"{imminent[0].currency} {imminent[0].name} in {int(imminent[0].minutes)} min")
    elif ev_pre:
        level = "MODERATE"; reasons.append(f"{ev_pre[0].currency} {ev_pre[0].name} in {int(ev_pre[0].minutes)} min (two-sided outcome)")
    post = [e for e in result["events"] if e.phase == "post" and e.minutes < 60]
    if post and level == "LOW":
        level = "MODERATE"; reasons.append(f"{post[0].currency} {post[0].name} released {int(post[0].minutes)} min ago")
    nx = sum(1 for n in allnames if recs[n][3600].status == "EXTREME VOLATILITY")
    if nx >= max(2, len(allnames) // 4):
        level = "HIGH"; reasons.append(f"extreme volatility in {nx} instruments")
    elif nx:
        level = level if level == "HIGH" else "MODERATE"; reasons.append(f"extreme volatility in {nx} instrument(s)")
    bad = sum(1 for n in allnames if recs[n][3600].status in ("STALE", "UNAVAILABLE"))
    if bad >= max(2, len(allnames) // 4):
        level = "HIGH"; reasons.append(f"{bad} instruments with stale/missing data")
    elif bad and level == "LOW":
        level = "MODERATE"; reasons.append(f"{bad} instrument(s) with stale/missing data")
    L.append(f"RISK: {level}" + (" — " + "; ".join(reasons) if reasons else " — no event or data-quality flags"))
    L.append("")
    L.append("Directional filter for your own PO3 signals only. UAMIS does not trade, and a preference is not a guaranteed signal. Not financial advice.")
    return "\n".join(L)
