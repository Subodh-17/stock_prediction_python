"""Coverage gaps found by the V20 coverage run: news_feed.py (28%) parsed untrusted XML with no tests; trade.py (0%) is the CLI that can act
on a signal. Both are tested at their real boundaries; no network and no broker object is ever touched."""
import os
import sys
from datetime import timezone

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
import news_feed as nf


# =============================================================================== news_feed
RSS = b"""<rss><channel>
<item><title>Alpha beats  estimates</title><link>https://ex.test/a</link><pubDate>Fri, 09 Oct 2026 10:00:00 GMT</pubDate></item>
<item><title>No date here</title><link>https://ex.test/b</link></item>
<item><title></title><link>https://ex.test/c</link></item>
<item><title>No link</title></item>
</channel></rss>"""
ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Atom item</title><link href="https://ex.test/atom"/>
<updated>2026-10-08T08:30:00+05:30</updated></entry></feed>"""


def test_feed_configuration_is_parsed_and_unsafe_schemes_are_dropped(monkeypatch):
    monkeypatch.setattr(config, "NEWS_FEEDS", "Pub A|https://a.test/rss?s={symbol}, |http://b.test/feed ,Bad|javascript:alert(1),File|file:///etc/passwd,justtext")
    feeds = nf.configured_feeds()
    assert [f["name"] for f in feeds] == ["Pub A", "b.test"] and all(f["url"].startswith(("http://", "https://")) for f in feeds)
    monkeypatch.setattr(config, "NEWS_FEEDS", "")
    assert nf.configured_feeds() == []


def test_rss_items_keep_publisher_title_link_and_honest_timestamps():
    items = nf.parse_feed(RSS, "Pub", "AAPL")
    assert [i["title"] for i in items] == ["Alpha beats  estimates", "No date here"]           # empty title / missing link are skipped
    a, b = items
    assert a["publisher"] == "Pub" and a["url"] == "https://ex.test/a" and a["symbol"] == "AAPL" and a["relevance"] is None
    assert a["published_ts"] == "2026-10-09T10:00:00+00:00" and b["published_ts"] is None       # a missing date is None, never invented
    assert a["retrieved_ts"] and a["retrieved_ts"] != a["published_ts"]


def test_atom_links_and_timezone_offsets_are_handled_and_normalised_to_utc():
    (item,) = nf.parse_feed(ATOM, "Atom Pub")
    assert item["url"] == "https://ex.test/atom" and item["published_ts"] == "2026-10-08T03:00:00+00:00"


@pytest.mark.parametrize("hostile", [b'<!DOCTYPE r [<!ENTITY x "y">]><rss/>', b"<rss/><!DOCTYPE lol>", b"<?xml version='1.0'?><!ENTITY a 'b'><rss/>"])
def test_doctype_and_entity_declarations_are_refused(hostile):
    with pytest.raises(ValueError, match="DOCTYPE|ENTITY"):
        nf.parse_feed(hostile, "P")


def test_oversized_and_malformed_feeds_are_rejected():
    with pytest.raises(ValueError, match="too large"):
        nf.parse_feed(b"<rss>" + b"x" * (nf.MAX_BYTES + 1) + b"</rss>", "P")
    import xml.etree.ElementTree as ET
    with pytest.raises(ET.ParseError):
        nf.parse_feed(b"<rss><item></rss>", "P")


def test_duplicates_are_removed_by_normalised_title_or_url():
    items = [{"title": "Same Story!", "url": "https://x/1"}, {"title": "same   story", "url": "https://y/2"},
             {"title": "Different", "url": "https://x/1"}, {"title": "Other", "url": "https://z/3"}]
    assert [i["title"] for i in nf.dedupe(items)] == ["Same Story!", "Other"]


def test_with_no_feeds_the_result_is_an_honest_empty_state_not_headlines(monkeypatch):
    monkeypatch.setattr(config, "NEWS_FEEDS", "")
    r = nf.get_news("AAPL")
    assert r["status"] == "not_configured" and r["items"] == [] and "No headlines are generated" in r["message"]


def test_get_news_orders_by_publication_time_isolates_a_failing_feed_and_quotes_the_symbol(monkeypatch):
    monkeypatch.setattr(config, "NEWS_FEEDS", "Good|https://g.test/rss?s={symbol},Broken|https://b.test/rss?s={symbol}")
    seen = []
    def fetcher(url):
        seen.append(url)
        if "b.test" in url:
            raise RuntimeError("HTTP 500 key=AIzaSyNEWSLEAK00000000000000000000000")
        return RSS
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "AIzaSyNEWSLEAK00000000000000000000000")
    r = nf.get_news("M&M.NS", fetcher=fetcher)
    assert r["status"] == "ok" and [i["title"] for i in r["items"]] == ["Alpha beats  estimates", "No date here"]    # dated first, undated last
    assert any("M%26M.NS" in u for u in seen)                                                                   # symbol is URL-quoted
    assert len(r["errors"]) == 1 and r["errors"][0]["feed"] == "Broken" and "AIzaSyNEWSLEAK" not in str(r["errors"])
    assert "not complete market coverage" in r["coverage_note"]


def test_status_distinguishes_all_feeds_failing_from_feeds_with_no_items_and_respects_the_limit(monkeypatch):
    monkeypatch.setattr(config, "NEWS_FEEDS", "A|https://a.test/rss")
    assert nf.get_news("X", fetcher=lambda u: (_ for _ in ()).throw(OSError("down")))["status"] == "error"
    assert nf.get_news("X", fetcher=lambda u: b"<rss><channel></channel></rss>")["status"] == "empty"
    many = b"<rss><channel>" + b"".join(f"<item><title>T{i}</title><link>https://e.test/{i}</link></item>".encode() for i in range(30)) + b"</channel></rss>"
    assert len(nf.get_news("X", limit=5, fetcher=lambda u: many)["items"]) == 5


# =============================================================================== trade.py CLI
@pytest.fixture
def cli(monkeypatch, capsys):
    import trade
    calls = {"pipeline": 0, "decision": []}
    def no_pipeline(**k):
        calls["pipeline"] += 1
        return {"final_ensemble": {"predicted_pct_change": 1.0, "final_predicted_price": 101.0, "current_price": 100.0}}
    def decision(**k):
        calls["decision"].append(k)
        return {"executed": False, "reason": "test double"}
    monkeypatch.setattr(trade.pipeline, "run_pipeline", no_pipeline)
    monkeypatch.setattr(trade, "execute_trade_decision", decision)
    def go(*argv):
        monkeypatch.setattr(sys, "argv", ["trade.py", *argv])
        return trade.main()
    return go, calls, capsys


def test_live_mode_is_refused_before_any_prediction_when_live_trading_is_not_enabled(cli, monkeypatch):
    go, calls, cap = cli
    monkeypatch.setattr(config, "ENABLE_LIVE_TRADING", False)
    assert go("--mode", "live", "--broker", "upstox", "--ticker", "AAPL") is None
    assert calls["pipeline"] == 0 and calls["decision"] == []
    assert "Live trading stays disabled" in cap.readouterr().out


def test_live_mode_also_needs_an_explicit_real_broker_even_when_enabled(cli, monkeypatch):
    go, calls, cap = cli
    monkeypatch.setattr(config, "ENABLE_LIVE_TRADING", True)
    for argv in (("--mode", "live"), ("--mode", "live", "--broker", "paper")):
        assert go(*argv) is None
    assert calls["pipeline"] == 0 and calls["decision"] == []
    assert "requires --broker upstox or --broker groww" in cap.readouterr().out


def test_the_default_is_paper_mode_without_auto_confirm(cli):
    go, calls, _ = cli
    out = go("--ticker", "AAPL", "--skip-llm")
    assert out == {"executed": False, "reason": "test double"} and calls["pipeline"] == 1
    kw = calls["decision"][0]
    assert kw["mode"] == "paper" and kw["auto_confirm"] is False and kw["broker_name"] is None


def test_auto_confirm_is_forwarded_only_when_explicitly_given_and_modes_are_restricted(cli, monkeypatch):
    go, calls, _ = cli
    go("--auto-confirm")
    assert calls["decision"][0]["auto_confirm"] is True
    import trade
    monkeypatch.setattr(sys, "argv", ["trade.py", "--mode", "yolo"])
    with pytest.raises(SystemExit):
        trade.parse_args()
    monkeypatch.setattr(sys, "argv", ["trade.py", "--source", "scrape"])
    with pytest.raises(SystemExit):
        trade.parse_args()
