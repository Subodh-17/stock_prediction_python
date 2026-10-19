import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

import config
import market_data as md
from market_data import (classify_freshness as cf, resolve_instrument, MarketDataService, QuoteProvider,
                         ProviderError, RateLimited, call_with_backoff, SingleFlight, TTLCache)

NOW = datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc)


def iso(sec_ago):
    return (NOW - timedelta(seconds=sec_ago)).isoformat()


# ---- instruments: same ticker on different exchanges must not be conflated
def test_nse_and_bse_listings_are_distinct_instruments():
    a, b = resolve_instrument("NSE:RELIANCE"), resolve_instrument("BSE:RELIANCE")
    assert a.key != b.key and a.yahoo_symbol == "RELIANCE.NS" and b.yahoo_symbol == "RELIANCE.BO"
    assert resolve_instrument("RELIANCE.NS").key == a.key


def test_bare_symbol_is_never_guessed_to_be_indian():
    i = resolve_instrument("tcs")
    assert i.exchange == "US/UNSPECIFIED" and i.yahoo_symbol == "TCS"


@pytest.mark.parametrize("bad", ["", None, "A B;DROP", "../etc", "X" * 40, "FOO:BAR", "<script>"])
def test_invalid_symbols_rejected(bad):
    with pytest.raises(md.InvalidSymbol):
        resolve_instrument(bad)


# ---- freshness policy
def test_live_requires_entitlement_and_fresh_source_timestamp():
    ok = cf(price=10, source_ts=iso(5), data_type="realtime", delay_seconds=0, supplies_realtime=True, now=NOW)
    assert ok["freshness"] == "live" and ok["realtime_eligible"]
    old = cf(price=10, source_ts=iso(config.LIVE_MAX_AGE_SECONDS + 60), data_type="realtime", delay_seconds=0, supplies_realtime=True, now=NOW)
    assert old["freshness"] == "stale" and not old["realtime_eligible"]


def test_provider_claim_of_realtime_without_entitlement_is_not_live():
    r = cf(price=10, source_ts=iso(1), data_type="realtime", delay_seconds=0, supplies_realtime=False, now=NOW)
    assert r["freshness"] != "live" and not r["realtime_eligible"]


def test_end_of_day_is_never_live_regardless_of_recent_retrieval():
    r = cf(price=10, source_ts=iso(3600), data_type="eod", delay_seconds=None, supplies_realtime=True, now=NOW)
    assert r["freshness"] == "end_of_day" and not r["realtime_eligible"]


def test_unknown_delay_is_unverified_not_guessed():
    r = cf(price=10, source_ts=iso(30), data_type="unknown", delay_seconds=None, supplies_realtime=False, now=NOW)
    assert r["freshness"] == "unverified"


def test_missing_timestamp_is_unverified_and_missing_price_unavailable():
    assert cf(price=10, source_ts=None, data_type="realtime", delay_seconds=None, supplies_realtime=True, now=NOW)["freshness"] == "unverified"
    for bad in (None, 0, -1, float("nan")):
        assert cf(price=bad, source_ts=iso(1), data_type="eod", delay_seconds=None, supplies_realtime=False, now=NOW)["freshness"] == "unavailable"


def test_delayed_goes_stale_after_expected_delay_plus_grace():
    fresh = cf(price=1, source_ts=iso(900 + 60), data_type="delayed", delay_seconds=900, supplies_realtime=False, now=NOW)
    stale = cf(price=1, source_ts=iso(900 + config.DELAYED_STALE_AFTER_SECONDS + 60), data_type="delayed", delay_seconds=900, supplies_realtime=False, now=NOW)
    assert fresh["freshness"] == "delayed" and not fresh["is_stale"]
    assert stale["freshness"] == "stale" and stale["is_stale"]


def test_future_source_timestamp_is_not_trusted():
    r = cf(price=1, source_ts=(NOW + timedelta(hours=2)).isoformat(), data_type="realtime", delay_seconds=0, supplies_realtime=True, now=NOW)
    assert r["freshness"] == "unverified" and "source_timestamp_in_future" in r["flags"]


def test_real_providers_declare_honest_capabilities():
    assert md.YahooProvider.supplies_realtime is False       # unofficial endpoint: never "Live"


# ---- service: fallback, outage, coalescing, caching
class Fake(QuoteProvider):
    def __init__(self, name, raw=None, exc=None, realtime=False, exchanges=(), delay=0.0):
        self.name, self._raw, self._exc, self.supplies_realtime = name, raw, exc, realtime
        self.covers_exchanges, self.calls, self.delay = exchanges, 0, delay
        super().__init__()

    def fetch_raw(self, inst):
        self.calls += 1
        time.sleep(self.delay)
        if self._exc:
            raise self._exc
        return dict(self._raw)


def raw(price=100.0, ts=None, dtype="delayed", delay=900.0):
    return {"price": price, "open": price, "high": price, "low": price, "prev_close": price - 1, "volume": 10,
            "currency": "USD", "source_ts": ts or datetime.now(timezone.utc) - timedelta(seconds=950),
            "data_type": dtype, "delay_seconds": delay, "session": "regular", "timezone": "UTC"}


def test_fallback_to_lower_freshness_is_visible():
    live = Fake("rt", exc=ProviderError("token expired", kind="auth"), realtime=True)
    slow = Fake("yahoo", raw=raw())
    q = MarketDataService([live, slow]).get_quote("AAPL", force=True)
    assert q.provider == "yahoo" and q.freshness == "delayed"
    assert q.downgraded_from == "rt" and "fallback_from_rt" in q.quality_flags
    assert [t["outcome"] for t in q.selection_trail] == ["auth", "ok"]


def test_no_eligible_provider_returns_unavailable_not_a_number():
    q = MarketDataService([Fake("upstox_only", raw=raw(), exchanges=("NSE",))]).get_quote("AAPL")
    assert q.freshness == "unavailable" and q.price is None and "No quote provider" in q.error


def test_all_providers_failing_reports_reasons():
    svc = MarketDataService([Fake("a", exc=ProviderError("boom A")), Fake("b", exc=ProviderError("boom B"))])
    q = svc.get_quote("AAPL", force=True)
    assert q.freshness == "unavailable" and q.price is None and "boom A" in q.error and "boom B" in q.error


def test_outage_serves_cached_quote_marked_cached_not_fresh():
    p = Fake("yahoo", raw=raw())
    svc = MarketDataService([p])
    good = svc.get_quote("MSFT", force=True)
    assert good.price == 100.0
    p._exc = ProviderError("down")
    out = svc.get_quote("MSFT", force=True)
    assert out.from_cache and "served_cached_during_outage" in out.quality_flags and out.error


def test_cache_hit_avoids_upstream_call_and_force_bypasses():
    p = Fake("yahoo", raw=raw())
    svc = MarketDataService([p])
    svc.get_quote("AAPL"); svc.get_quote("AAPL")
    assert p.calls == 1
    svc.get_quote("AAPL", force=True)
    assert p.calls == 2


def test_concurrent_identical_requests_share_one_upstream_call():
    p = Fake("yahoo", raw=raw(), delay=0.3)
    svc = MarketDataService([p])
    out = []
    ts = [threading.Thread(target=lambda: out.append(svc.get_quote("AAPL", force=True))) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert p.calls == 1 and len(out) == 6 and svc.flight.coalesced >= 5


def test_invalid_ohlc_is_flagged():
    r = raw(); r.update(open=100, high=90, low=95, price=100)
    q = MarketDataService([Fake("yahoo", raw=r)]).get_quote("AAPL", force=True)
    assert "invalid_ohlc_relationship" in q.quality_flags


def test_repeated_failures_open_circuit_and_stop_calling_provider():
    p = Fake("flaky", exc=ProviderError("x"))
    svc = MarketDataService([p])
    for _ in range(4):
        svc.get_quote("AAPL", force=True)
    n = p.calls
    q = svc.get_quote("AAPL", force=True)
    assert p.calls == n and q.selection_trail[0]["outcome"] == "skipped"


# ---- retries
def test_backoff_honours_retry_after_and_is_bounded():
    sleeps, n = [], {"c": 0}
    def fn():
        n["c"] += 1
        raise RateLimited("429", retry_after=3)
    with pytest.raises(RateLimited):
        call_with_backoff(fn, retries=2, sleep=sleeps.append)
    assert n["c"] == 3 and all(3 <= s < 3.3 for s in sleeps)


def test_non_retryable_error_is_not_retried():
    n = {"c": 0}
    def fn():
        n["c"] += 1
        raise ProviderError("bad symbol", retryable=False)
    with pytest.raises(ProviderError):
        call_with_backoff(fn, retries=3, sleep=lambda s: None)
    assert n["c"] == 1


def test_ttl_cache_is_bounded_and_expires():
    c = TTLCache(maxsize=2)
    c.set("a", 1, 100); c.set("b", 2, 100); c.set("c", 3, 100)
    assert c.get("a") is None and c.get("c")[0] == 3
    c.set("z", 9, 0.01); time.sleep(0.03)
    assert c.get("z") is None


# ---- history + partial-bar handling
def test_history_normalises_and_reports_quality(monkeypatch):
    from tests.v17_helpers import fixture_bars
    import data_sources.yfinance_source as ys
    calls = {"n": 0}
    monkeypatch.setattr(ys, "fetch", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), fixture_bars(120))[1])
    svc = MarketDataService([])
    h = svc.get_history("NSE:TCS", "1y")
    assert h["instrument"]["yahoo_symbol"] == "TCS.NS" and h["adjusted"] and h["rows"] == 120 and len(h["bars"]) == 120
    assert h["freshness"] == "historical" and h["provider"] == "yahoo"
    svc.get_history("NSE:TCS", "1y")
    assert calls["n"] == 1


def test_history_rejects_bad_period_and_interval():
    svc = MarketDataService([])
    with pytest.raises(ValueError):
        svc.get_history("AAPL", "1y; rm -rf")
    with pytest.raises(ValueError):
        svc.get_history("AAPL", "1y", interval="1m")


def test_partial_bar_dropped_only_when_session_open():
    import forecast_service as fs
    from tests.v17_helpers import fixture_bars
    inst = resolve_instrument("AAPL")
    df = fixture_bars(30, start="2026-09-01")
    last = df.index[-1]
    now = datetime(last.year, last.month, last.day, 15, 0, tzinfo=timezone.utc)      # 11:00 New York: open
    out, dropped = fs.drop_partial_bar(df, inst, "regular", now=now)
    assert dropped == str(last)[:10] and len(out) == len(df) - 1
    out2, d2 = fs.drop_partial_bar(df, inst, "closed", now=now)
    assert d2 is None and len(out2) == len(df)
