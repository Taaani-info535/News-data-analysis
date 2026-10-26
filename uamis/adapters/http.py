"""HTTP helper: timeouts, retries with exponential backoff, per-host rate limiting. Honors 429 Retry-After."""
from __future__ import annotations
import logging
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

log = logging.getLogger("uamis.http")


class HttpError(Exception):
    pass


class HttpClient:
    def __init__(self, timeout=20, retries=3, backoff=2.0, min_interval=0.5, user_agent="UAMIS/0.1", session=None, sleep=time.sleep):
        self.timeout, self.retries, self.backoff, self.min_interval = timeout, retries, backoff, min_interval
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._last: Dict[str, float] = {}
        self._sleep = sleep

    @classmethod
    def from_config(cls, cfg, **kw):
        h = cfg.raw.get("http", {})
        return cls(h.get("timeout_seconds", 20), h.get("retries", 3), h.get("backoff_seconds", 2.0),
                   h.get("min_interval_seconds", 0.5), h.get("user_agent", "UAMIS/0.1"), **kw)

    def _throttle(self, host: str):
        now = time.monotonic()
        wait = self._last.get(host, 0) + self.min_interval - now
        if wait > 0:
            self._sleep(wait)
        self._last[host] = time.monotonic()

    def request(self, method: str, url: str, **kw) -> requests.Response:
        host = urlparse(url).netloc
        last_exc: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            self._throttle(host)
            try:
                r = self.session.request(method, url, timeout=self.timeout, **kw)
                if r.status_code == 429 or r.status_code >= 500:
                    ra = r.headers.get("Retry-After")
                    delay = float(ra) if ra and ra.replace(".", "", 1).isdigit() else self.backoff * (2 ** attempt)
                    last_exc = HttpError(f"HTTP {r.status_code} from {host}")
                    if attempt < self.retries:
                        log.warning("retryable HTTP %s from %s; sleeping %.1fs", r.status_code, host, delay)
                        self._sleep(min(delay, 120))
                        continue
                    break
                if r.status_code >= 400:
                    raise HttpError(f"HTTP {r.status_code} from {host}")   # non-retryable
                return r
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                if attempt < self.retries:
                    self._sleep(self.backoff * (2 ** attempt))
        raise HttpError(f"request failed after retries: {type(last_exc).__name__}: {last_exc}")

    def get_json(self, url: str, **kw) -> Any:
        try:
            return self.request("GET", url, **kw).json()
        except ValueError as e:
            raise HttpError(f"invalid JSON from {urlparse(url).netloc}") from e

    def get_text(self, url: str, **kw) -> str:
        return self.request("GET", url, **kw).text
