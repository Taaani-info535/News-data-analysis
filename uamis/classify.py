"""Rules-based event/news classification and *theoretical* impact mapping.
Theory is NOT a forecast: it states the textbook direction; observed reaction is analysed separately."""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

# (regex on lower-cased title, category, hawkish_if_above)  hawkish_if_above: True => higher-than-forecast is USD/hawkish-positive,
# False => higher-than-forecast is dovish (e.g. unemployment), None => no directional rule (speeches, statements)
EVENT_RULES: List[Tuple[str, str, Optional[bool]]] = [
    (r"core pce|pce price|core pce price", "PCE", True),
    (r"\bcore cpi|cpi", "CPI", True),
    (r"core ppi|\bppi\b", "PPI", True),
    (r"non-?farm|nfp|nonfarm", "NFP", True),
    (r"unemployment rate", "UNEMPLOYMENT", False),
    (r"unemployment claims|jobless claims", "CLAIMS", False),
    (r"average hourly earnings|employment cost", "WAGES", True),
    (r"adp", "ADP", True),
    (r"jolts|job openings", "JOLTS", True),
    (r"fomc.*(minutes)|meeting minutes|monetary policy meeting accounts", "FED_MINUTES", None),
    (r"federal funds rate|fomc statement|fomc economic projections|rate decision|official bank rate|main refinancing|"
     r"cash rate|overnight rate|policy rate|monetary policy statement|boj policy|snb|rba rate|boc rate|rbnz", "FED_RATE", True),
    (r"fomc press conference|press conference", "FED_SPEECH", None),
    (r"speaks|speech|testif|chair .* powell|governor", "FED_SPEECH", None),
    (r"retail sales", "RETAIL", True),
    (r"\bgdp\b", "GDP", True),
    (r"pmi|ism ", "PMI", True),
    (r"consumer sentiment|consumer confidence|michigan", "SENTIMENT", True),
    (r"durable goods|industrial production|factory orders|trade balance|housing|building permits|home sales", "OTHER_DATA", True),
]
FOMC_HINT = re.compile(r"fomc|federal funds|fed ")

NEWS_RULES: List[Tuple[str, str]] = [
    (r"federal open market committee.*(statement|decision)|fomc statement|implementation note|monetary policy", "FED_RATE"),
    (r"minutes of the federal open market|fomc minutes", "FED_MINUTES"),
    (r"speech|testimony|remarks by|statement by", "FED_SPEECH"),
    (r"consumer price index|\bcpi\b", "CPI"),
    (r"personal income and outlays|\bpce\b", "PCE"),
    (r"employment situation|nonfarm|payroll", "NFP"),
    (r"sanction|war\b|missile|invasion|ceasefire|military|embargo|tariff|attack", "GEOPOLITICS"),
]

SUPPORTED_CATEGORIES = sorted({c for _, c, _ in EVENT_RULES} | {c for _, c in NEWS_RULES} | {"OTHER"})
LEARNABLE_CATEGORIES = [c for c in SUPPORTED_CATEGORIES if c != "OTHER"]


def classify_event(name: str) -> Tuple[str, Optional[bool]]:
    n = name.lower()
    for rx, cat, hawk in EVENT_RULES:
        if re.search(rx, n):
            return cat, hawk
    return "OTHER", None


def hawkish_rule(category: str, name: str = "") -> Optional[bool]:
    c, h = classify_event(name) if name else (category, None)
    if name and c == category:
        return h
    for _, cat, hk in EVENT_RULES:
        if cat == category:
            return hk
    return None


def classify_news(headline: str) -> str:
    h = headline.lower()
    for rx, cat in NEWS_RULES:
        if re.search(rx, h):
            return cat
    return "OTHER"


def relevant_assets(currency: str, assets: Dict[str, Dict]) -> List[str]:
    """Assets possibly affected by an event in `currency`. USD events touch every USD-quoted/based asset, indices and crypto.
    Non-USD events touch assets where that currency is base or quote."""
    out = []
    for a, m in assets.items():
        if currency in (m.get("base"), m.get("quote")):
            out.append(a)
        elif currency == "USD" and (m.get("base") in ("IDX", "XAU", "XAG", "BTC", "ETH") or m.get("quote") == "USD"):
            out.append(a)
    return sorted(set(out))


def surprise_sign(actual: Optional[float], forecast: Optional[float], tol_rel: float = 1e-9) -> Optional[int]:
    if actual is None or forecast is None:
        return None
    if abs(actual - forecast) <= tol_rel * max(1.0, abs(forecast)):
        return 0
    return 1 if actual > forecast else -1


def surprise_rel(actual: Optional[float], forecast: Optional[float], previous: Optional[float]) -> Optional[float]:
    if actual is None or forecast is None:
        return None
    scale = max(abs(forecast), abs(previous) if previous is not None else 0.0, 1e-9)
    return max(-1.0, min(1.0, (actual - forecast) / scale))


def theoretical_impact(event_currency: str, hawkish_if_above: Optional[bool], sign: Optional[int], asset_meta: Dict) -> Tuple[str, str]:
    """Return (BULLISH/BEARISH/NEUTRAL/N/A, explanation). Textbook transmission only: hawkish currency surprise -> that
    currency stronger; USD strength -> pressure on gold/indices/crypto. Observed price action may disagree."""
    if hawkish_if_above is None or sign is None:
        return "N/A", "no directional rule for this event type or no actual-vs-forecast available"
    if sign == 0:
        return "NEUTRAL", "actual equals forecast"
    hawkish = (sign > 0) == hawkish_if_above   # True => currency-strengthening surprise
    base, quote = asset_meta.get("base"), asset_meta.get("quote")
    if event_currency == base and base not in ("IDX", "XAU", "XAG", "BTC", "ETH"):
        eff = 1 if hawkish else -1
        why = f"{event_currency} {'strength' if hawkish else 'weakness'} implied"
    elif event_currency == quote:
        eff = -1 if hawkish else 1
        why = f"{event_currency} {'strength' if hawkish else 'weakness'} implied (it is the quote currency)"
    elif event_currency == "USD":   # USD event, USD not base/quote => gold/indices/crypto side
        eff = -1 if hawkish else 1
        why = f"USD/yield {'upside' if hawkish else 'downside'} implied; historically pressures/supports non-yielding and risk assets"
    else:
        return "N/A", "event currency not related to this asset"
    return ("BULLISH" if eff > 0 else "BEARISH"), why
