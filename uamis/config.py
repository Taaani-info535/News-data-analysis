"""Configuration loading. Secrets come ONLY from environment variables."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

DEFAULT_CONFIG_PATH = Path(os.environ.get("UAMIS_CONFIG", "config/config.yaml"))


@dataclass
class Config:
    raw: Dict[str, Any]

    # ---- convenience accessors -------------------------------------------------
    @property
    def db_path(self) -> str:
        return os.environ.get("UAMIS_DB", self.raw.get("db_path", "state/uamis.sqlite"))

    @property
    def model_dir(self) -> str:
        return os.environ.get("UAMIS_MODEL_DIR", self.raw.get("model_dir", "state/models"))

    @property
    def assets(self) -> Dict[str, Dict[str, Any]]:
        return self.raw["assets"]

    @property
    def horizons(self) -> List[int]:
        return [int(h) for h in self.raw["horizons_minutes"]]

    # ---- Forex list: loaded verbatim from the user's file, never guessed --------
    def forex_status(self) -> Tuple[List[str], str]:
        """Return (pairs, status_message). pairs is [] unless the file holds exactly expected_count valid pairs."""
        fx = self.raw.get("forex") or {}
        path = Path(fx.get("pairs_file", "config/forex_pairs.txt"))
        expected = int(fx.get("expected_count", 28))
        if not path.exists():
            return [], f"FOREX DISABLED: {path} not found. Provide your scanner's {expected} pairs."
        pairs: List[str] = []
        bad: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            t = line.split("#", 1)[0].strip().upper().replace("/", "")
            if not t:
                continue
            if re.fullmatch(r"[A-Z]{6}", t):
                pairs.append(t)
            else:
                bad.append(t)
        if bad:
            return [], f"FOREX DISABLED: invalid pair names {bad[:5]} (use 6 letters, e.g. EURUSD)."
        if len(set(pairs)) != len(pairs):
            return [], "FOREX DISABLED: duplicate pairs in forex_pairs.txt."
        if len(pairs) != expected:
            return [], f"FOREX DISABLED: found {len(pairs)} pairs, expected exactly {expected}. Not guessing the rest."
        return pairs, f"FOREX OK: {len(pairs)} pairs loaded from {path}"

    def all_assets(self) -> Dict[str, Dict[str, Any]]:
        """Non-forex assets plus (only if the list is valid) all configured forex pairs."""
        if getattr(self, "_assets_cache", None) is not None:
            return self._assets_cache
        out = dict(self.assets)
        pairs, _ = self.forex_status()
        fx = self.raw.get("forex") or {}
        tmpl = fx.get("yahoo_symbol_template", "{pair}=X")
        for pr in pairs:
            out[pr] = {"source": "yahoo", "symbol": tmpl.format(pair=pr), "base": pr[:3], "quote": pr[3:],
                       "cost_bps": fx.get("cost_bps_default", 1.5), "class": "forex"}
        self._assets_cache = out
        return out

    def g(self, *path, default=None):
        cur: Any = self.raw
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    # Secrets: never stored in raw config, never logged.
    @property
    def telegram_token(self) -> str:
        return os.environ.get("TELEGRAM_BOT_TOKEN", "")

    @property
    def telegram_chat_id(self) -> str:
        return os.environ.get("TELEGRAM_CHAT_ID", "")


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw)
