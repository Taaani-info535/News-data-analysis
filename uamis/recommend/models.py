"""Data structures for the recommendation engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LONG, SHORT, WAIT, AWAY = "LONG PREFERENCE", "SHORT PREFERENCE", "NEUTRAL / WAIT", "STAY AWAY"
DIRECTIONS = (LONG, SHORT, WAIT, AWAY)
CODE = {LONG: "L", SHORT: "S", WAIT: "W", AWAY: "X"}
HZ_NAME = {900: "15m", 3600: "1h", 14400: "4h", 86400: "24h"}

# evidence kinds: price evidence may CONFIRM or CONTRADICT but can never create a preference on its own
FUNDAMENTAL, MODEL, PRICE = "fundamental", "model", "price"


@dataclass
class Evidence:
    key: str                      # macro | tone | rates | risk | cryptomkt | model | reaction | price | strength
    label: str
    value: Optional[float]        # -1..+1, positive = bullish for the INSTRUMENT (price up). None = data unavailable
    weight: float                 # nominal weight for this horizon
    text: str
    kind: str = FUNDAMENTAL
    applicable: bool = True       # False = not relevant for this instrument (dropped; does not reduce coverage)
    data_ts: Optional[int] = None

    @property
    def available(self) -> bool:
        return self.applicable and self.value is not None


@dataclass
class Recommendation:
    instrument: str
    horizon_s: int
    direction: str
    score: float                              # -100..+100 (bullish positive)
    confidence: str                           # LOW | MODERATE | HIGH | n/a
    status: str                               # UNVALIDATED | VALIDATED | UNAVAILABLE | STALE | MARKET CLOSED | EVENT RISK | PENDING | ...
    supporting: List[str] = field(default_factory=list)
    contradicting: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    data_ts: Optional[int] = None
    freshness: str = "UNAVAILABLE"            # RECENT | DELAYED | STALE | CLOSED | UNAVAILABLE
    event: Optional[str] = None
    stage: str = "SCAN"                       # SCAN | PRE-NEWS BIAS | POST-NEWS | POST-NEWS PENDING
    created_ts: int = 0
    coverage: float = 0.0
    group: str = ""
    kind: str = "forex"
    thr: Optional[float] = None               # sideways threshold frozen for later evaluation (abs log-return)
    vol_pct: Optional[float] = None
    validated: bool = False
    components: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["horizon"] = HZ_NAME.get(self.horizon_s, f"{self.horizon_s}s")
        return d


@dataclass
class CurrencyOutlook:
    currency: str
    score: float                  # -100..+100 fundamental (macro surprises, tone, USD yields)
    direction: str                # BULLISH | BEARISH | NEUTRAL
    confidence: str
    reasons: List[str] = field(default_factory=list)
    strength_z: Optional[float] = None    # price-derived currency strength (1h-4h), context only
    data_ts: Optional[int] = None


@dataclass
class EventCtx:
    event_id: str
    name: str
    currency: str
    t0: int
    importance: int
    phase: str                    # pre | post | post_pending
    minutes: float                # minutes to release (pre) or since release (post)
    forecast_raw: str = ""
    previous_raw: str = ""
    actual_raw: str = ""
    affected: List[str] = field(default_factory=list)
