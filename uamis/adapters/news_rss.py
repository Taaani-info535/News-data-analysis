"""RSS news adapter (official primary sources by default). Dedupes by normalised headline hash across sources."""
from __future__ import annotations
import calendar
import hashlib
import re
from typing import List

try:
    import feedparser  # type: ignore
except ImportError:  # stdlib fallback keeps basic RSS/Atom support available offline
    feedparser = None
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

from ..classify import classify_news
from ..models import NewsItem


def norm_headline(h: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", h.lower()).strip()


def parse_feed_text(text: str, source: str, official: bool, now_ts: int, max_age_s: int) -> List[NewsItem]:
    if feedparser is not None:
        d = feedparser.parse(text)
        entries = []
        for e in d.entries:
            pt = e.get("published_parsed") or e.get("updated_parsed")
            pub = calendar.timegm(pt) if pt else None
            entries.append((e.get("title") or "", e.get("link", ""), pub))
    else:
        entries = _stdlib_entries(text)
    out: List[NewsItem] = []
    for title, link, pub in entries:
        title = (title or "").strip()
        if not title or pub is None:
            continue                                  # no publication time => cannot store honestly
        if pub > now_ts + 300 or now_ts - pub > max_age_s:
            continue
        nid = hashlib.sha1(norm_headline(title).encode()).hexdigest()[:16]
        out.append(NewsItem(nid, int(pub), source, title, link or "", classify_news(title), [], official, now_ts))
    return out


def _stdlib_entries(text: str):
    """Small RSS/Atom parser fallback; never guesses missing publication dates."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    def local(tag):
        return tag.rsplit("}", 1)[-1].lower()
    entries = []
    for node in root.iter():
        if local(node.tag) not in ("item", "entry"):
            continue
        fields = {}
        link = ""
        for child in list(node):
            key = local(child.tag)
            if key == "link":
                link = child.attrib.get("href", "") or (child.text or "").strip()
            elif key in ("title", "pubdate", "published", "updated", "date"):
                fields[key] = ("".join(child.itertext()) or "").strip()
        date_text = fields.get("pubdate") or fields.get("published") or fields.get("updated") or fields.get("date")
        pub = None
        if date_text:
            try:
                dt = parsedate_to_datetime(date_text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                pub = int(dt.timestamp())
            except (TypeError, ValueError, OverflowError):
                try:
                    dt = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    pub = int(dt.timestamp())
                except (TypeError, ValueError, OverflowError):
                    pub = None
        entries.append((fields.get("title", ""), link, pub))
    return entries


def fetch_news(http, feeds: List[dict], now_ts: int, max_age_hours: float) -> List[NewsItem]:
    items: List[NewsItem] = []
    for f in feeds:
        try:
            text = http.get_text(f["url"])
            items.extend(parse_feed_text(text, f["name"], bool(f.get("official")), now_ts, int(max_age_hours * 3600)))
        except Exception as e:      # one failing feed must not kill the run
            import logging
            logging.getLogger("uamis.news").warning("feed %s failed: %s", f.get("name"), type(e).__name__)
    return items
