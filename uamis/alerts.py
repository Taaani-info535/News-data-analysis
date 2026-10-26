"""Telegram delivery (official Bot API), message builders, deduplication and cooldowns. Token is never logged."""
from __future__ import annotations
import hashlib
import logging
import time
from typing import Callable, Dict, List, Optional

import requests

from .classify import surprise_sign
from .models import Event, Prediction
from .util import ts_to_iso

log = logging.getLogger("uamis.alerts")
HZ = {900: "15M", 3600: "1H", 14400: "4H", 86400: "24H"}


class TelegramError(Exception):
    pass


class TelegramSender:
    def __init__(self, token: str, chat_id: str, timeout: int = 20, retries: int = 3, sleep: Callable = time.sleep, post=requests.post):
        self.token, self.chat_id, self.timeout, self.retries, self._sleep, self._post = token, chat_id, timeout, retries, sleep, post

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str, max_chars: int = 3500) -> None:
        for chunk in split_message(text, max_chars):
            self._send_chunk(chunk)

    def _send_chunk(self, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        last = "unknown"
        for attempt in range(self.retries + 1):
            try:
                r = self._post(url, json={"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True}, timeout=self.timeout)
            except requests.RequestException as e:
                last = type(e).__name__          # never include URL (contains token)
                self._sleep(2 ** attempt); continue
            if r.status_code == 200:
                return
            if r.status_code == 429:
                try:
                    ra = float(r.json().get("parameters", {}).get("retry_after", 2))
                except Exception:
                    ra = 2.0
                last = "rate limited"; self._sleep(min(ra, 60)); continue
            if r.status_code >= 500:
                last = f"server {r.status_code}"; self._sleep(2 ** attempt); continue
            raise TelegramError(f"Telegram rejected message (HTTP {r.status_code}); check bot token / chat id")
        raise TelegramError(f"Telegram send failed after retries ({last})")


def split_message(text: str, limit: int) -> List[str]:
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            out.append(cur); cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur.rstrip("\n"))
    return out


def digest(parts: List[str]) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def dispatch(store, cfg, sender: Optional[TelegramSender], event_id: str, stage: str, text: str, material: List[str], now_ts: int,
             dry_run: bool = False, out=print, cooldown_s: Optional[int] = None) -> str:
    """Send at most once per (event, stage, material-content). Same stage with changed content is rate-limited by cooldown.
    Records the alert only after successful delivery so a failed run retries instead of silently dropping."""
    dg = digest(material)
    key = f"{event_id}:{stage}:{dg}"
    last = store.alert_last(event_id, stage)
    if last is not None and last["digest"] == dg:
        return "duplicate"
    cooldown = int(cfg.g("alerts", "cooldown_minutes", default=180)) * 60 if cooldown_s is None else int(cooldown_s)
    if last is not None and now_ts - last["sent_ts"] < cooldown:
        return "cooldown"
    if dry_run or sender is None or not sender.configured:
        out(f"--- [{'DRY RUN' if dry_run else 'TELEGRAM NOT CONFIGURED'}] {stage} {event_id} ---\n{text}\n")
        return "dry_run"
    sender.send(text, int(cfg.g("alerts", "max_message_chars", default=3500)))
    store.alert_record(key, event_id, stage, now_ts, dg)
    return "sent"


# ------------------------------------------------------------------ message builders (plain text) ---------------------------
def _cell(p: Optional[Prediction]) -> str:
    if p is None:
        return "n/a"
    if p.status_label == "INSUFFICIENT DATA":
        return "INSUF"
    a = {"UP": "UP", "DOWN": "DOWN", "SIDEWAYS": "SIDE", "UNCERTAIN": "UNC"}[p.argmax]
    if p.status_label == "VALIDATED" and p.probs and p.argmax != "UNCERTAIN":
        return f"{a} {p.probs[p.argmax]:.0%}"
    return a + "*"


def pair_table(cfg, preds: Dict[str, Dict[int, Prediction]], pairs: List[str], fx_msg: str, relevant: List[str]) -> List[str]:
    """Compact table that ALWAYS lists every configured forex pair (or states why Forex is disabled)."""
    if not pairs:
        return [f"FOREX: {fx_msg}"]
    hs = [900, 3600, 14400, 86400]
    lines = ["FOREX (all %d configured pairs)" % len(pairs), "PAIR   | 15M    | 1H     | 4H     | 24H    | D"]
    for pr in pairs:
        if pr not in relevant:
            lines.append(f"{pr} | not affected by this event's currency")
            continue
        row = preds.get(pr, {})
        d = row.get(3600) or next(iter(row.values()), None)
        dcode = "-" if d is None else ("STAY" if d.decision == "STAY AWAY" else ("WAIT" if d.decision.startswith("WAIT") else "OK"))
        lines.append(f"{pr} | " + " | ".join(f"{_cell(row.get(h)):<6}" for h in hs) + f" | {dcode}")
    lines.append("* experimental/not validated  INSUF = insufficient data  D: STAY=stay away, WAIT=wait for release")
    return lines


def _asset_block(asset: str, row: Dict[int, Prediction]) -> List[str]:
    p1 = row.get(3600)
    if p1 is None:
        return [f"{asset}: no analysis (market data unavailable)"]
    lines = [f"{asset} - 1H: {p1.argmax} [{p1.status_label}]"]
    if p1.probs:
        lines.append("  P(up/down/side): " + " / ".join(f"{p1.probs[k]:.0%}" for k in ("UP", "DOWN", "SIDEWAYS"))
                     + ("" if p1.status_label == "VALIDATED" else "  (NOT calibrated-validated)"))
    h = p1.hist
    lines.append("  Historical sample: " + (f"n={h['n']} (of {h['n_same_category']} same-category)" if h.get("status") == "OK" else f"INSUFFICIENT HISTORICAL DATA (n={h.get('n_same_category', 0)})"))
    lines.append(f"  Risk: {p1.risk_level} | Decision: {p1.decision}")
    if p1.reasons:
        lines.append("  Why: " + "; ".join(p1.reasons[:3]))
    return lines


def format_upcoming(ev: Event, assets: List[str], now_ts: int) -> str:
    mins = max(0, (ev.t0 - now_ts) // 60)
    return (f"AI MARKET INTELLIGENCE\nUPCOMING EVENT: {ev.currency} {ev.name}\nRelease: {ts_to_iso(ev.t0)} (in {mins // 60}h {mins % 60}m)\n"
            f"Importance: {'*' * ev.importance}  Forecast: {ev.forecast_raw or 'not provided'}  Previous: {ev.previous_raw or 'not provided'}\n"
            f"Possibly affected: {', '.join(assets[:12])}{'...' if len(assets) > 12 else ''}\nA pre-event analysis follows closer to the release. No forecast of the result is made.")


def format_pre(cfg, ev: Event, preds: Dict[str, Dict[int, Prediction]], scen: Dict[str, List[str]], focus: List[str], pairs, fx_msg, relevant, now_ts: int) -> str:
    mins = max(0, (ev.t0 - now_ts) // 60)
    L = ["AI MARKET INTELLIGENCE", f"UPCOMING EVENT: {ev.currency} {ev.name}", f"Stage: PRE-NEWS | Release: {ts_to_iso(ev.t0)} | in {mins} min",
         f"Forecast: {ev.forecast_raw or 'n/a'} | Previous: {ev.previous_raw or 'n/a'} (result unknown)", ""]
    for a in focus:
        if a in preds:
            L += _asset_block(a, preds[a]); 
            for s in scen.get(a, []):
                L.append("  " + s)
            L.append("")
    L += pair_table(cfg, preds, pairs, fx_msg, relevant)
    L.append("Forecasts are probabilistic, not guarantees. This is not a trade signal.")
    return "\n".join(L)


def format_post(cfg, ev: Event, preds, focus, pairs, fx_msg, relevant, theory: Dict[str, tuple], context: List[str], now_ts: int, note: str = "") -> str:
    sg = surprise_sign(ev.actual, ev.forecast)
    sd = {1: "ABOVE forecast", -1: "BELOW forecast", 0: "IN LINE with forecast", None: "surprise not computable (forecast missing)"}[sg]
    L = ["AI MARKET INTELLIGENCE", f"RELEASED: {ev.currency} {ev.name}", f"Stage: POST-NEWS | Release: {ts_to_iso(ev.t0)} | Source timestamp seen: {ts_to_iso(ev.actual_seen_at)}",
         f"Actual: {ev.actual_raw} | Forecast: {ev.forecast_raw or 'n/a'} | Previous: {ev.previous_raw or 'n/a'} -> {sd}", ""]
    for a in focus:
        th = theory.get(a)
        if a in preds:
            L += _asset_block(a, preds[a])
        if th:
            L.append(f"  Theory: {th[0]} ({th[1]}) - textbook impact, not the observed reaction")
        L.append("")
    L += context
    L += pair_table(cfg, preds, pairs, fx_msg, relevant)
    if note:
        L.append(note)
    L.append("Conflicting or weak signals are reported as UNCERTAIN, not forced. Not a trade signal.")
    return "\n".join(L)


def format_news(item, assets: List[str]) -> str:
    v = "OFFICIAL SOURCE" if item.verified else "UNVERIFIED REPORT"
    return (f"AI MARKET INTELLIGENCE\nBREAKING ({v}): {item.headline}\nSource: {item.source} | Published: {ts_to_iso(item.published)}\n"
            f"Category: {item.category} | Possibly affected: {', '.join(assets[:10])}\nURL: {item.url}\n"
            "Unscheduled news: no historical-model forecast is generated; watch volatility and wait for confirmation.")
