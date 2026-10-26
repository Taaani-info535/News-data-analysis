"""Provider symbol validation. Malformed symbols are rejected BEFORE any request is made (never sent to the provider)."""
from __future__ import annotations
import re
from typing import Dict, List

CRYPTO_BASES = {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT", "LTC", "TRX", "MATIC", "TON"}
_YAHOO_FX = re.compile(r"^[A-Z]{6}=X$")
_YAHOO_GENERIC = re.compile(r"^\^?[A-Z0-9]{1,10}([=.\-][A-Z0-9]{1,6}){0,3}$")
_BINANCE = re.compile(r"^[A-Z0-9]{5,15}$")


def asset_kind(meta: Dict) -> str:
    """forex | crypto | metal | index | dollar_index | other. Uses config `class`, else infers from base currency."""
    c = (meta.get("class") or "").lower()
    if c in ("forex", "crypto", "metal", "index", "dollar_index"):
        return c
    base = meta.get("base")
    if base in CRYPTO_BASES:
        return "crypto"
    if base in ("XAU", "XAG"):
        return "metal"
    if base == "IDX":
        return "index"
    return "other"


def validate_asset(name: str, meta: Dict) -> List[str]:
    """Return a list of problems (empty = valid)."""
    errs: List[str] = []
    src, sym = meta.get("source"), meta.get("symbol")
    if src not in ("yahoo", "binance"):
        errs.append(f"unknown source {src!r}")
        return errs
    if not isinstance(sym, str) or not sym.strip() or sym != sym.strip():
        return [f"empty or padded symbol {sym!r}"]
    kind = asset_kind(meta)
    if src == "yahoo":
        if kind == "forex":
            if not _YAHOO_FX.match(sym):
                errs.append(f"yahoo forex symbol {sym!r} must look like EURUSD=X")
            elif sym[:6] != name:
                errs.append(f"symbol {sym!r} does not match pair {name}")
        elif not _YAHOO_GENERIC.match(sym):
            errs.append(f"yahoo symbol {sym!r} has unexpected characters")
    else:
        if not _BINANCE.match(sym):
            errs.append(f"binance symbol {sym!r} must be upper-case letters/digits, e.g. BTCUSDT")
        elif not sym.endswith(("USDT", "USDC", "BUSD")):
            errs.append(f"binance symbol {sym!r} is not a USD-proxy pair (USDT/USDC/BUSD)")
    if not meta.get("base") or not meta.get("quote"):
        errs.append("missing base/quote")
    return errs


def validate_all(assets: Dict[str, Dict]) -> Dict[str, List[str]]:
    return {n: e for n, m in assets.items() if (e := validate_asset(n, m))}
