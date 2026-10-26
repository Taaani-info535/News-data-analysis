"""Economic-calendar adapter (community Forex Factory JSON feed; unofficial, current-week only).
Never fabricates: blank forecast/previous/actual stay None."""
from __future__ import annotations
import hashlib
import logging
from typing import List

from ..classify import classify_event
from ..models import Event
from ..util import parse_iso_to_ts, parse_value, utcnow_ts

log = logging.getLogger("uamis.calendar")
IMPACT = {"high": 3, "medium": 2, "low": 1}


def make_event_id(currency: str, name: str, t0: int) -> str:
    return hashlib.sha1(f"{currency}|{name.strip().lower()}|{t0}".encode()).hexdigest()[:16]


def parse_feed(items: list, source: str, now_ts: int | None = None) -> List[Event]:
    now_ts = now_ts or utcnow_ts()
    out: List[Event] = []
    skipped = 0
    for it in items or []:
        try:
            impact = IMPACT.get(str(it.get("impact", "")).strip().lower())
            if impact is None:        # holidays etc.
                skipped += 1
                continue
            name, cur = str(it["title"]).strip(), str(it["country"]).strip().upper()
            t0 = parse_iso_to_ts(str(it["date"]))
        except (KeyError, ValueError, TypeError):
            skipped += 1
            continue
        cat, _ = classify_event(name)
        act_raw = str(it.get("actual", "") or "")
        out.append(Event(
            event_id=make_event_id(cur, name, t0), name=name, currency=cur, t0=t0, importance=impact, category=cat,
            forecast_raw=str(it.get("forecast", "") or ""), previous_raw=str(it.get("previous", "") or ""), actual_raw=act_raw,
            forecast=parse_value(it.get("forecast")), previous=parse_value(it.get("previous")), actual=parse_value(act_raw),
            source=source, first_seen=now_ts, updated_at=now_ts, actual_seen_at=now_ts if parse_value(act_raw) is not None else None))
    if skipped:
        log.info("calendar: skipped %d unusable rows", skipped)
    return out


def fetch_calendar(http, urls: List[str], now_ts: int | None = None) -> List[Event]:
    events: List[Event] = []
    for u in urls:
        data = http.get_json(u)
        if not isinstance(data, list):
            raise ValueError("unexpected calendar payload type")
        events.extend(parse_feed(data, source=u, now_ts=now_ts))
    return events
