"""SQLite persistence (single file; idempotent upserts so failed/duplicate runs are safe)."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from .models import Event, NewsItem, Prediction

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles(asset TEXT, res INTEGER, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY(asset,res,ts));
CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY, name TEXT, currency TEXT, t0 INTEGER, importance INTEGER, category TEXT,
  forecast_raw TEXT, previous_raw TEXT, actual_raw TEXT, forecast REAL, previous REAL, actual REAL, source TEXT,
  first_seen INTEGER, updated_at INTEGER, actual_seen_at INTEGER);
CREATE INDEX IF NOT EXISTS ix_events_t0 ON events(t0);
CREATE TABLE IF NOT EXISTS news(news_id TEXT PRIMARY KEY, published INTEGER, source TEXT, headline TEXT, url TEXT, category TEXT,
  assets TEXT, verified INTEGER, first_seen INTEGER);
CREATE TABLE IF NOT EXISTS predictions(pred_id TEXT PRIMARY KEY, event_id TEXT, asset TEXT, stage TEXT, horizon_s INTEGER,
  created_ts INTEGER, anchor_ts INTEGER, probs TEXT, baseline TEXT, argmax TEXT, status_label TEXT, model_version TEXT,
  hist_n INTEGER, hist TEXT, risk_level TEXT, decision TEXT, reasons TEXT, features TEXT, thr REAL, vol_pct REAL, category TEXT,
  ret_actual REAL, outcome_label TEXT, eval_status TEXT DEFAULT 'PENDING', evaluated_ts INTEGER, high_vol INTEGER);
CREATE INDEX IF NOT EXISTS ix_pred_eval ON predictions(eval_status, anchor_ts);
CREATE TABLE IF NOT EXISTS alerts(alert_key TEXT PRIMARY KEY, event_id TEXT, stage TEXT, sent_ts INTEGER, digest TEXT);
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS recommendations(rec_id TEXT PRIMARY KEY, created_ts INTEGER, instrument TEXT, horizon_s INTEGER, direction TEXT,
  score REAL, confidence TEXT, status TEXT, stage TEXT, event_name TEXT, thr REAL, vol_pct REAL, grp TEXT, components TEXT,
  ret_actual REAL, outcome_label TEXT, eval_status TEXT DEFAULT 'PENDING', evaluated_ts INTEGER);
CREATE INDEX IF NOT EXISTS ix_rec_eval ON recommendations(eval_status, created_ts);
"""


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        self.db.close()

    # ---- candles ----------------------------------------------------------------
    def upsert_candles(self, asset: str, res: int, rows: Iterable[tuple]) -> int:
        """rows: (ts, open, high, low, close, volume). Later data for same ts overwrites (corrections)."""
        data = [(asset, res, int(r[0]), r[1], r[2], r[3], r[4], r[5]) for r in rows]
        self.db.executemany("INSERT OR REPLACE INTO candles VALUES(?,?,?,?,?,?,?,?)", data)
        self.db.commit()
        return len(data)

    def load_candles(self, asset: str, res: int, since: Optional[int] = None, until: Optional[int] = None) -> List[sqlite3.Row]:
        q = "SELECT ts,open,high,low,close,volume FROM candles WHERE asset=? AND res=?"
        a: list = [asset, res]
        if since is not None:
            q += " AND ts>=?"; a.append(since)
        if until is not None:
            q += " AND ts<=?"; a.append(until)
        return self.db.execute(q + " ORDER BY ts", a).fetchall()

    def last_candle_ts(self, asset: str, res: int) -> Optional[int]:
        r = self.db.execute("SELECT MAX(ts) m FROM candles WHERE asset=? AND res=?", (asset, res)).fetchone()
        return r["m"]

    def prune_candles(self, res: int, older_than_ts: int):
        self.db.execute("DELETE FROM candles WHERE res=? AND ts<?", (res, older_than_ts))
        self.db.commit()

    # ---- events -----------------------------------------------------------------
    def upsert_event(self, e: Event) -> str:
        """Insert or update. Never overwrites a known actual with blank. Returns 'new'|'updated'|'same'."""
        cur = self.db.execute("SELECT * FROM events WHERE event_id=?", (e.event_id,)).fetchone()
        if cur is None:
            self.db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (e.event_id, e.name, e.currency, e.t0, e.importance, e.category, e.forecast_raw, e.previous_raw, e.actual_raw,
                 e.forecast, e.previous, e.actual, e.source, e.first_seen, e.updated_at, e.actual_seen_at))
            self.db.commit()
            return "new"
        changed = False
        fields = {"forecast_raw": e.forecast_raw, "previous_raw": e.previous_raw, "forecast": e.forecast, "previous": e.previous,
                  "importance": e.importance, "category": e.category}
        if e.actual is not None and (cur["actual"] is None or cur["actual"] != e.actual):
            fields.update(actual=e.actual, actual_raw=e.actual_raw, actual_seen_at=e.actual_seen_at or e.updated_at)
        sets, vals = [], []
        for k, v in fields.items():
            if v is None or v == "":
                continue
            if cur[k] != v:
                sets.append(f"{k}=?"); vals.append(v); changed = True
        if changed:
            sets.append("updated_at=?"); vals.append(e.updated_at)
            self.db.execute(f"UPDATE events SET {','.join(sets)} WHERE event_id=?", vals + [e.event_id])
            self.db.commit()
            return "updated"
        return "same"

    def _row_event(self, r) -> Event:
        return Event(**{k: r[k] for k in r.keys()})

    def get_event(self, event_id: str) -> Optional[Event]:
        r = self.db.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        return self._row_event(r) if r else None

    def events_between(self, t_from: int, t_to: int, min_importance: int = 1) -> List[Event]:
        rows = self.db.execute("SELECT * FROM events WHERE t0>=? AND t0<? AND importance>=? ORDER BY t0",
                               (t_from, t_to, min_importance)).fetchall()
        return [self._row_event(r) for r in rows]

    def all_events_before(self, t: int) -> List[Event]:
        rows = self.db.execute("SELECT * FROM events WHERE t0<? ORDER BY t0", (t,)).fetchall()
        return [self._row_event(r) for r in rows]

    # ---- news -------------------------------------------------------------------
    def insert_news(self, n: NewsItem) -> bool:
        cur = self.db.execute("INSERT OR IGNORE INTO news VALUES(?,?,?,?,?,?,?,?,?)",
            (n.news_id, n.published, n.source, n.headline, n.url, n.category, json.dumps(n.assets), int(n.verified), n.first_seen))
        self.db.commit()
        return cur.rowcount > 0

    # ---- predictions ------------------------------------------------------------
    def insert_prediction(self, p: Prediction) -> bool:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO predictions(pred_id,event_id,asset,stage,horizon_s,created_ts,anchor_ts,probs,baseline,argmax,"
            "status_label,model_version,hist_n,hist,risk_level,decision,reasons,features,thr,vol_pct,category) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.pred_id, p.event_id, p.asset, p.stage, p.horizon_s, p.created_ts, p.anchor_ts, json.dumps(p.probs),
             json.dumps(p.baseline), p.argmax, p.status_label, p.model_version, p.hist_n, json.dumps(p.hist), p.risk_level,
             p.decision, json.dumps(p.reasons), json.dumps(p.features), p.thr, p.vol_pct, p.category))
        self.db.commit()
        return cur.rowcount > 0

    def retry_insufficient_prediction(self, p: Prediction) -> bool:
        """Replace a previous missing-data placeholder if a later run has real features.

        A deterministic prediction id normally makes the ledger immutable. This narrowly-scoped
        exception prevents a temporary data outage at first scan from permanently freezing an
        event/horizon as INSUFFICIENT DATA. Existing non-placeholder forecasts are never changed.
        """
        if p.status_label == "INSUFFICIENT DATA":
            return False
        cur = self.db.execute(
            "UPDATE predictions SET created_ts=?,anchor_ts=?,probs=?,baseline=?,argmax=?,status_label=?,"
            "model_version=?,hist_n=?,hist=?,risk_level=?,decision=?,reasons=?,features=?,thr=?,vol_pct=?,category=? "
            "WHERE pred_id=? AND status_label='INSUFFICIENT DATA' AND eval_status='PENDING'",
            (p.created_ts, p.anchor_ts, json.dumps(p.probs), json.dumps(p.baseline), p.argmax, p.status_label,
             p.model_version, p.hist_n, json.dumps(p.hist), p.risk_level, p.decision, json.dumps(p.reasons),
             json.dumps(p.features), p.thr, p.vol_pct, p.category, p.pred_id))
        self.db.commit()
        return cur.rowcount > 0

    def pending_predictions(self) -> List[sqlite3.Row]:
        return self.db.execute("SELECT * FROM predictions WHERE eval_status='PENDING'").fetchall()

    def set_outcome(self, pred_id: str, ret: Optional[float], label: Optional[str], status: str, ts: int, high_vol: int):
        self.db.execute("UPDATE predictions SET ret_actual=?, outcome_label=?, eval_status=?, evaluated_ts=?, high_vol=? WHERE pred_id=?",
                        (ret, label, status, ts, high_vol, pred_id))
        self.db.commit()

    def evaluated_predictions(self) -> List[sqlite3.Row]:
        return self.db.execute("SELECT * FROM predictions WHERE eval_status='EVALUATED'").fetchall()

    # ---- recommendations ledger (directional LONG/SHORT preferences only; evaluated after their horizon) ------------------
    def insert_recommendation(self, r: dict) -> bool:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO recommendations(rec_id,created_ts,instrument,horizon_s,direction,score,confidence,status,stage,event_name,"
            "thr,vol_pct,grp,components) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["rec_id"], r["created_ts"], r["instrument"], r["horizon_s"], r["direction"], r["score"], r["confidence"], r["status"],
             r["stage"], r.get("event_name"), r.get("thr"), r.get("vol_pct"), r.get("grp"), json.dumps(r.get("components", {}))))
        self.db.commit()
        return cur.rowcount > 0

    def pending_recommendations(self) -> List[sqlite3.Row]:
        return self.db.execute("SELECT * FROM recommendations WHERE eval_status='PENDING'").fetchall()

    def set_recommendation_outcome(self, rec_id: str, ret: Optional[float], label: Optional[str], status: str, ts: int):
        self.db.execute("UPDATE recommendations SET ret_actual=?, outcome_label=?, eval_status=?, evaluated_ts=? WHERE rec_id=?",
                        (ret, label, status, ts, rec_id))
        self.db.commit()

    def evaluated_recommendations(self, horizon_s: Optional[int] = None) -> List[sqlite3.Row]:
        q, a = "SELECT * FROM recommendations WHERE eval_status='EVALUATED'", []
        if horizon_s is not None:
            q += " AND horizon_s=?"; a.append(horizon_s)
        return self.db.execute(q, a).fetchall()

    # ---- alerts / kv ------------------------------------------------------------
    def alert_last(self, event_id: str, stage: str) -> Optional[sqlite3.Row]:
        return self.db.execute("SELECT * FROM alerts WHERE event_id=? AND stage=? ORDER BY sent_ts DESC LIMIT 1",
                               (event_id, stage)).fetchone()

    def alert_record(self, key: str, event_id: str, stage: str, ts: int, digest: str):
        self.db.execute("INSERT OR REPLACE INTO alerts VALUES(?,?,?,?,?)", (key, event_id, stage, ts, digest))
        self.db.commit()

    def kv_get(self, k: str, default: Optional[str] = None) -> Optional[str]:
        r = self.db.execute("SELECT value FROM kv WHERE key=?", (k,)).fetchone()
        return r["value"] if r else default

    def kv_set(self, k: str, v: str):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", (k, v))
        self.db.commit()
