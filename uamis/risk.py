"""Independent STAY AWAY / risk filter. Separates: directional forecast, confidence, volatility risk, and 'suitable to act'.
All thresholds come from config (defaults are NOT universally validated). A forecast is never a trade recommendation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RiskInput:
    stage: str
    event_importance: int
    status_label: str                       # VALIDATED / EXPERIMENTAL / LOW CONFIDENCE / INSUFFICIENT DATA
    probs: Dict[str, float]
    ml_argmax: Optional[str] = None
    ml_top_prob: Optional[float] = None
    study_argmax: Optional[str] = None
    study_top_prob: Optional[float] = None
    vol_pct: Optional[float] = None
    staleness_s: Optional[int] = None
    max_staleness_s: int = 7200
    other_event_minutes: Optional[float] = None
    deteriorated: bool = False
    late_fraction: float = 0.0
    data_missing: bool = False
    released: bool = False


def assess(ri: RiskInput, rc: dict) -> Tuple[str, str, List[str]]:
    """Returns (risk_level, decision, reasons). Decision is one of: STAY AWAY, WAIT FOR RELEASE, CONDITIONS ACCEPTABLE (not a signal)."""
    hard: List[str] = []
    soft: List[str] = []
    if ri.data_missing or ri.vol_pct is None:
        hard.append("market data missing")
    if ri.staleness_s is not None and ri.staleness_s > ri.max_staleness_s:
        hard.append(f"market data stale ({ri.staleness_s // 60} min old)")
    if ri.status_label == "INSUFFICIENT DATA":
        hard.append("insufficient historical data")
    elif ri.status_label != "VALIDATED":
        hard.append(f"model {ri.status_label.lower()}: not validated out-of-sample")
    if ri.deteriorated:
        hard.append("live model performance deteriorated vs baseline")
    top = max(ri.probs.values()) if ri.probs else None
    if top is not None and top < float(rc["min_top_probability"]):
        hard.append("low confidence (no outcome clearly favoured)")
    if (ri.ml_argmax and ri.study_argmax and ri.ml_argmax != ri.study_argmax
            and (ri.ml_top_prob or 0) >= float(rc["disagreement_min_prob"]) and (ri.study_top_prob or 0) >= float(rc["disagreement_min_prob"])):
        hard.append(f"conflicting signals (model {ri.ml_argmax} vs history {ri.study_argmax})")
    if ri.vol_pct is not None and ri.vol_pct >= float(rc["vol_pct_high"]):
        hard.append(f"extreme volatility (hourly vol at {ri.vol_pct:.0%} percentile)")
    if ri.other_event_minutes is not None and ri.other_event_minutes <= float(rc["imminent_event_minutes"]):
        soft.append(f"another important event within {int(ri.other_event_minutes)} min")
    if ri.late_fraction > 0.25:
        hard.append("analysis delayed: a large part of the horizon had already elapsed")
    if ri.stage == "pre" and ri.event_importance >= 3:
        soft.append("high-importance release pending (outcome unknown)")

    reasons = hard + soft
    if hard or (ri.vol_pct is not None and ri.vol_pct >= float(rc["vol_pct_high"])):
        level = "HIGH" if (ri.vol_pct is not None and ri.vol_pct >= float(rc["vol_pct_high"])) or ri.stage == "pre" or ri.data_missing or (ri.staleness_s or 0) > ri.max_staleness_s else "MEDIUM"
    else:
        level = "MEDIUM" if soft else "LOW"
    if ri.stage == "pre" and level != "HIGH" and ri.event_importance >= 3:
        level = "HIGH"
    if hard:
        decision = "STAY AWAY"
    elif ri.stage == "pre" and not ri.released:
        decision = "WAIT FOR RELEASE"
    else:
        decision = "CONDITIONS ACCEPTABLE (not a trade signal)"
    return level, decision, reasons
