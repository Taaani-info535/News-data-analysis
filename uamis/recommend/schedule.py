"""Stage scheduling for PRE-NEWS / POST-NEWS reports with catch-up semantics.
GitHub cron is best-effort, so a stage is 'due' once its trigger time has passed and its window is still open. If several stages are
due at once (late/skipped runs), only the LATEST one is reported and the earlier ones are marked superseded - never a burst of messages."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from ..models import Event


@dataclass
class Stage:
    name: str
    kind: str          # pre | post
    trigger_ts: int


@dataclass
class DueItem:
    event: Event
    key: str                       # stage key to mark processed (pre_60 / post_15 / post_pending)
    kind: str                      # pre | post | pending
    superseded: List[str] = field(default_factory=list)


def proc_key(event_id: str, stage: str) -> str:
    return f"rec_stage:{event_id}:{stage}"


def stages_open(ev: Event, now_ts: int, rc: dict) -> List[Stage]:
    """All stages whose trigger has passed and whose window is open, ordered by trigger time."""
    out: List[Stage] = []
    if now_ts < ev.t0:
        for m in sorted((int(x) for x in rc["pre_checks_minutes"]), reverse=True):
            trig = ev.t0 - m * 60
            if trig <= now_ts:
                out.append(Stage(f"pre_{m}", "pre", trig))
    elif now_ts <= ev.t0 + int(rc["post_window_minutes"]) * 60:
        for m in sorted(int(x) for x in rc["post_checks_minutes"]):
            trig = ev.t0 + m * 60
            if trig <= now_ts:
                out.append(Stage(f"post_{m}", "post", trig))
    return out


def due_item(store, ev: Event, now_ts: int, rc: dict) -> "DueItem | None":
    stages = stages_open(ev, now_ts, rc)
    if not stages:
        return None
    last = stages[-1]
    superseded = [s.name for s in stages[:-1] if store.kv_get(proc_key(ev.event_id, s.name)) is None]
    if last.kind == "pre":
        if store.kv_get(proc_key(ev.event_id, last.name)) is not None:
            return None
        return DueItem(ev, last.name, "pre", superseded)
    if ev.actual is None:                       # released but actual unavailable => one 'pending' notice, real stage stays open
        if store.kv_get(proc_key(ev.event_id, "post_pending")) is not None:
            return None
        return DueItem(ev, "post_pending", "pending", [])
    if store.kv_get(proc_key(ev.event_id, last.name)) is not None:
        return None
    return DueItem(ev, last.name, "post", superseded)


def mark_processed(store, item: DueItem, now_ts: int):
    for k in [item.key] + list(item.superseded):
        store.kv_set(proc_key(item.event.event_id, k), str(now_ts))
