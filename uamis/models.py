"""Plain data containers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LABELS = ["UP", "DOWN", "SIDEWAYS"]


@dataclass
class Event:
    event_id: str
    name: str
    currency: str            # e.g. USD
    t0: int                  # release time, UTC epoch seconds
    importance: int          # 1 low, 2 medium, 3 high
    category: str
    forecast_raw: str = ""
    previous_raw: str = ""
    actual_raw: str = ""
    forecast: Optional[float] = None
    previous: Optional[float] = None
    actual: Optional[float] = None
    source: str = ""
    first_seen: int = 0
    updated_at: int = 0
    actual_seen_at: Optional[int] = None


@dataclass
class NewsItem:
    news_id: str
    published: int
    source: str
    headline: str
    url: str
    category: str
    assets: List[str] = field(default_factory=list)
    verified: bool = False
    first_seen: int = 0


@dataclass
class Prediction:
    pred_id: str
    event_id: str
    asset: str
    stage: str               # pre | post
    horizon_s: int
    created_ts: int
    anchor_ts: int           # time the outcome window starts
    probs: Dict[str, float]  # UP/DOWN/SIDEWAYS (may be empty if INSUFFICIENT)
    baseline: Dict[str, float]
    argmax: str              # UP/DOWN/SIDEWAYS/UNCERTAIN
    status_label: str        # VALIDATED / EXPERIMENTAL / LOW CONFIDENCE / INSUFFICIENT DATA
    model_version: str
    hist_n: int
    hist: Dict
    risk_level: str
    decision: str
    reasons: List[str]
    features: Dict[str, float]
    thr: Optional[float]     # sideways threshold (abs log-return) fixed at prediction time
    vol_pct: Optional[float]
    category: str = ""
    drivers: List[str] = field(default_factory=list)
