"""OPTIONAL language-model helper (OpenAI-compatible endpoint). Disabled by default; the system never depends on it.
It may only *rephrase/summarise* text already produced by the rules engine; it must not add facts, numbers or predictions."""
from __future__ import annotations
import logging
import os
from typing import Optional

log = logging.getLogger("uamis.llm")


def summarize(http, cfg, text: str) -> Optional[str]:
    c = cfg.raw.get("llm", {})
    key = os.environ.get("LLM_API_KEY", "")
    if not (c.get("enabled") and c.get("endpoint") and c.get("model") and key):
        return None
    try:
        r = http.request("POST", c["endpoint"], headers={"Authorization": f"Bearer {key}"}, json={
            "model": c["model"], "temperature": 0,
            "messages": [{"role": "system", "content": "Rewrite the report in at most 3 plain sentences. Do not add facts, numbers or predictions not present."},
                         {"role": "user", "content": text}]})
        out = r.json()["choices"][0]["message"]["content"].strip()
        return out or None
    except Exception as e:
        log.warning("LLM summary unavailable (%s); using rules-based text", type(e).__name__)
        return None
