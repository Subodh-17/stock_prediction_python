"""Mocked-transport behaviour tests. NOTHING here proves a real provider works; scripts/verify_providers.py does that."""
import json
import socket
from datetime import datetime, timedelta, timezone

import pytest
import requests

import config
import market_data as md
import provider_verification as pv
from ai_integrations import providers as P
from tests.v17_helpers import FakeResp, fixture_bars


# ---- documented Upstox timestamp format
def test_parse_ts_understands_upstox_epoch_milliseconds_string():
    ts = md._parse_ts("1697624972130")                                   # example value from Upstox's own docs
    assert ts == datetime.fromtimestamp(1697624972.130, tz=timezone.utc)
    assert md._parse_ts(1697624972130) == ts and md._parse_ts("1697624972") == datetime.fromtimestamp(1697624972, tz=timezone.utc)
    assert md._parse_ts("garbage") is None and md._parse_ts("") is None and md._parse_ts(-5) is None
    assert md._parse_ts("2026-10-09T10:00:00Z").tzinfo is not None


def _upstox_env(monkeypatch, response):
    from data_sources import upstox_source
    monkeypatch.setattr(config, "UPSTOX_CLIENT_ID", "client-id-for-test")
    monkeypatch.setattr(upstox_source, "_load_cached_token", lambda: "tok-123")
    calls = []
    monkeypatch.setattr(upstox_source, "_resolve_instrument_key", lambda sym, exchange="NSE": (calls.append((sym, exchange)), f"{exchange}_EQ|X")[1])
    monkeypatch.setattr(requests, "get", lambda *a, **k: (calls.append(("GET", k.get("params"), k["headers"]["Authorization"])), response)[1])
    return calls


def quote_body(ms_ago):
    t = int((datetime.now(timezone.utc) - timedelta(milliseconds=ms_ago)).timestamp() * 1000)
    return {"status": "success", "data": {"NSE_EQ:RELIANCE": {"last_price": 1500.5, "volume": 1000, "last_trade_time": str(t),
            "ohlc": {"open": 1490.0, "high": 1510.0, "low": 1488.0, "close": 1495.0}}}}


def test_upstox_adapter_classifies_by_the_documented_timestamp_and_never_better(monkeypatch):
    calls = _upstox_env(monkeypatch, FakeResp(200, quote_body(3000)))
    svc = md.MarketDataService([md.UpstoxQuoteProvider()])
    q = svc.get_quote("NSE:RELIANCE", force=True)
    assert q.freshness == "live" and q.realtime_eligible and q.exchange == "NSE" and q.currency == "INR"
    assert ("RELIANCE", "NSE") in calls and calls[-1][1] == {"instrument_key": "NSE_EQ|X"}
    old = _upstox_env(monkeypatch, FakeResp(200, quote_body(10 * 60 * 1000)))
    q2 = md.MarketDataService([md.UpstoxQuoteProvider()]).get_quote("NSE:RELIANCE", force=True)
    assert q2.freshness == "stale" and not q2.realtime_eligible          # illiquid/old last trade is never "live"


def test_bse_quotes_request_the_bse_instrument_list_not_nse(monkeypatch):
    calls = _upstox_env(monkeypatch, FakeResp(200, quote_body(1000)))
    md.MarketDataService([md.UpstoxQuoteProvider()]).get_quote("BSE:RELIANCE", force=True)
    assert ("RELIANCE", "BSE") in calls and ("RELIANCE", "NSE") not in calls


@pytest.mark.parametrize("status,headers,exc", [(401, {}, md.ProviderError), (403, {}, md.ProviderError),
                                                (429, {"Retry-After": "7"}, md.RateLimited), (503, {}, md.ProviderError)])
def test_upstox_error_statuses_are_classified_not_turned_into_quotes(monkeypatch, status, headers, exc):
    _upstox_env(monkeypatch, FakeResp(status, {"status": "error"}, headers))
    p = md.UpstoxQuoteProvider()
    with pytest.raises(exc) as e:
        p.fetch_raw(md.resolve_instrument("NSE:RELIANCE"))
    if status == 429:
        assert e.value.retry_after == 7.0
    q = md.MarketDataService([p]).get_quote("NSE:RELIANCE", force=True)
    assert q.price is None and q.freshness == "unavailable"


def test_legacy_resolver_passes_bse_for_bo_tickers_and_leaves_nse_calls_unchanged(monkeypatch):
    import data_sources as ds
    seen = []
    monkeypatch.setattr(ds.upstox_source, "fetch", lambda *a, **k: (seen.append((a, k)), fixture_bars(30))[1])
    monkeypatch.setattr(ds.groww_source, "fetch", lambda *a, **k: (seen.append((a, k)), fixture_bars(30))[1])
    ds.fetch_stock_data("RELIANCE.BO", source="upstox", verbose=False)
    ds.fetch_stock_data("RELIANCE.NS", source="upstox", verbose=False)
    ds.fetch_stock_data("RELIANCE", source="groww", verbose=False)
    ds.fetch_stock_data("tcs.bo", source="groww", verbose=False)
    assert seen[0][1] == {"exchange": "BSE"} and seen[1][1] == {} and seen[2][1] == {} and seen[3][1] == {"exchange": "BSE"}


def test_legacy_suffix_fallback_marks_the_resolved_ticker(monkeypatch):
    import data_sources as ds
    from data_sources.base import DataSourceError
    def fake(sym, *a, **k):
        if sym == "TCS":
            raise DataSourceError("no data")
        return fixture_bars(30)
    monkeypatch.setattr(ds.yfinance_source, "fetch", fake)
    df = ds.fetch_stock_data("TCS", source="yfinance", verbose=False)
    assert df.attrs["resolved_ticker"] == "TCS.NS"


# ---- AI transport honesty
def test_an_injected_transport_can_never_produce_a_verified_status(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "AIzaSyTESTKEY00000000000000000000")
    models = FakeResp(200, {"models": [{"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]}]})
    ok = FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
    p = P.GeminiProvider(lambda m, url, **k: models if "pageSize" in url else ok)
    r = p.test_connection()
    st = p.status()
    assert r["ok"] and r["transport"] == "TEST DOUBLE" and st["state"] == "mock_verified" and st["transport"] == "test double"
    assert P.GeminiProvider().real_transport is True and P.Registry().providers["gemini"].real_transport is True


# ---- verification records
@pytest.fixture
def blocked(monkeypatch):
    def refuse(*a, **k):
        raise OSError("no route")
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_unreachable_hosts_are_BLOCKED_never_PASS(blocked, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "AIzaSySECRETSECRETSECRET0000000000")
    monkeypatch.setattr(config, "UPSTOX_CLIENT_ID", "cid")
    from data_sources import upstox_source
    monkeypatch.setattr(upstox_source, "_load_cached_token", lambda: "tok")
    recs = pv.run_all(svc=md.MarketDataService([]), registry=P.Registry())
    by = {(r["provider"], r["endpoint_class"].split()[0]): r["outcome"] for r in recs}
    assert by[("yahoo", "history")] == "BLOCKED" and by[("yahoo", "quote")] == "BLOCKED" and by[("upstox", "quote")] == "BLOCKED"
    assert by[("gemini", "LLM")] == "BLOCKED"
    assert all(r["outcome"] in ("BLOCKED", "NOT_CONFIGURED") for r in recs) and not any(r["outcome"] == "PASS" for r in recs)
    assert by[("groq", "LLM")] == "NOT_CONFIGURED"
    dumped = json.dumps(recs)
    assert "SECRETSECRET" not in dumped and "tok" not in json.dumps([r["evidence"] for r in recs])
    path = pv.save(recs)
    assert pv.load()["records"] == recs and path.endswith("provider_verification.json")


def test_pass_requires_a_validated_real_response_not_just_a_reachable_host(monkeypatch):
    monkeypatch.setattr(pv, "reachable", lambda *a, **k: True)
    class Bad(md.QuoteProvider):
        name, supplies_realtime = "yahoo", False
        def fetch_raw(self, inst):
            return {"price": 10.0, "source_ts": None, "data_type": "unknown", "delay_seconds": None}   # no timestamp
    r = pv._check_quote("yahoo", Bad(), "AAPL", require_config=False)
    assert r["outcome"] == "FAIL" and r["evidence"]["freshness_category"] == "unverified"
    class Good(md.QuoteProvider):
        name, supplies_realtime = "yahoo", False
        def fetch_raw(self, inst):
            return {"price": 10.0, "open": 9.9, "high": 10.2, "low": 9.8, "source_ts": datetime.now(timezone.utc), "data_type": "delayed", "delay_seconds": 900.0}
    assert pv._check_quote("yahoo", Good(), "AAPL", require_config=False)["outcome"] == "PASS"


def test_health_endpoint_states_that_nothing_is_verified_when_no_records_exist(tmp_path, monkeypatch):
    import os, sys
    sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webapp")]
    import app as webapp
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    j = webapp.app.test_client().get("/api/v2/health").get_json()
    assert j["verification"]["records"] == [] and "NOTHING has been verified" in j["verification"]["note"]
