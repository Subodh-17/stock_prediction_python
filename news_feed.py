"""
news_feed.py -- headlines from RSS/Atom feeds YOU configure (NEWS_FEEDS). Nothing is invented:
with no feeds configured, or all failing, the result says so. Each item keeps the publisher's
title, link, publication time (None if the feed omitted it) and our retrieval time.
Feeds are fetched server-side; XML with a DOCTYPE is refused (entity-expansion guard); 1 MB cap.
"""
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import config
import security

MAX_BYTES = 1_000_000


def configured_feeds() -> list:
    feeds = []
    for part in [p.strip() for p in (config.NEWS_FEEDS or "").split(",") if p.strip()]:
        name, _, url = part.partition("|")
        if url and urlparse(url.strip()).scheme in ("http", "https"):
            feeds.append({"name": name.strip() or urlparse(url).netloc, "url": url.strip()})
    return feeds


def _ts(text):
    if not text:
        return None
    try:
        d = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).isoformat()


def parse_feed(xml_bytes: bytes, publisher: str, symbol: str = None) -> list:
    if len(xml_bytes) > MAX_BYTES:
        raise ValueError("feed too large")
    if re.search(rb"<!DOCTYPE|<!ENTITY", xml_bytes[:4096] + xml_bytes[-1024:], re.I):
        raise ValueError("feed contains a DOCTYPE/ENTITY declaration and was refused")
    root = ET.fromstring(xml_bytes)
    now = datetime.now(timezone.utc).isoformat()
    out = []
    for it in root.iter():
        tag = it.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        g = lambda n: next((c for c in it if c.tag.split("}")[-1] == n), None)
        title, link = g("title"), g("link")
        href = (link.get("href") if link is not None and link.get("href") else (link.text if link is not None else None))
        if title is None or not (title.text or "").strip() or not href:
            continue
        pub = next((e for e in (g("pubDate"), g("published"), g("updated")) if e is not None), None)
        out.append({"title": title.text.strip(), "publisher": publisher, "url": href.strip(),
                    "published_ts": _ts(pub.text if pub is not None else None), "retrieved_ts": now,
                    "symbol": symbol, "relevance": None})
    return out


def dedupe(items: list) -> list:
    seen, out = set(), []
    for i in items:
        k = re.sub(r"\W+", " ", i["title"].lower()).strip() or i["url"]
        if k in seen or i["url"] in seen:
            continue
        seen.update((k, i["url"]))
        out.append(i)
    return out


def get_news(symbol: str, limit: int = 20, fetcher=None) -> dict:
    import requests
    feeds = configured_feeds()
    if not feeds:
        return {"status": "not_configured", "items": [], "errors": [],
                "message": "No news feeds configured. Set NEWS_FEEDS in .env, e.g. "
                           "NEWS_FEEDS=Publisher|https://example.com/rss?s={symbol}. No headlines are generated."}
    items, errors = [], []
    for f in feeds:
        url = f["url"].replace("{symbol}", requests.utils.quote(symbol))
        try:
            body = fetcher(url) if fetcher else requests.get(
                url, timeout=config.NEWS_TIMEOUT_SECONDS, headers={"User-Agent": "stock-workstation/17"}).content
            items += parse_feed(body, f["name"], symbol)
        except Exception as e:  # noqa: BLE001
            errors.append({"feed": f["name"], "error": security.safe_exception_text(e)[:200]})
    items = dedupe(items)
    items.sort(key=lambda i: i["published_ts"] or "", reverse=True)
    status = "ok" if items else ("error" if errors else "empty")
    return {"status": status, "items": items[:limit], "errors": errors,
            "coverage_note": "Only the configured feeds are searched; this is not complete market coverage."}
