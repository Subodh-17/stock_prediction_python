"""Regression tests for Quant Workstation V21: Market-Data Reliability and Circuit Breaker Integrity."""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, timezone

import market_data as md


def test_no_data_query_does_not_trip_circuit_breaker():
    """An unknown/delisted symbol returning no_data must NOT count as an infrastructure outage
    or trip the circuit breaker for subsequent valid requests."""
    p = md.QuoteProvider()
    p.name = "test_provider"
    
    # 4 consecutive no_data queries
    for _ in range(4):
        try:
            raise md.ProviderError("Yahoo returned no data for TCS.", retryable=False, kind="no_data")
        except md.ProviderError as e:
            if e.kind == "no_data":
                p.health.record_no_data(str(e))
            else:
                p.health.fail(str(e))
                
    st = p.health.snapshot()
    assert st["failures"] == 0, "no_data queries must not increment infrastructure failures"
    assert st["consecutive_failures"] == 0
    assert st["no_data_queries"] == 4
    assert st["circuit_open"] is False, "Circuit must remain CLOSED after no_data queries"
    assert p.health.available is True


def test_market_data_service_no_data_does_not_block_valid_ticker():
    """Simulate querying 3 non-existent tickers followed by a valid ticker.
    The valid ticker must NOT be skipped due to an open circuit."""
    calls = []
    
    class FlakyProvider(md.QuoteProvider):
        name = "mock_provider"
        def fetch_raw(self, inst: md.Instrument) -> dict:
            calls.append(inst.symbol)
            if inst.symbol == "UNKNOWN":
                raise md.ProviderError(f"No data for {inst.symbol}", retryable=False, kind="no_data")
            return {
                "price": 150.0, "open": 149.0, "high": 151.0, "low": 148.0,
                "prev_close": 149.5, "volume": 1000000, "currency": "USD",
                "source_ts": datetime.now(timezone.utc).isoformat(), "session": "regular",
                "data_type": "delayed", "delay_seconds": 900.0, "timezone": "America/New_York",
            }
            
    prov = FlakyProvider()
    svc = md.MarketDataService([prov])
    
    # Query 4 times for UNKNOWN
    for _ in range(4):
        q = svc.get_quote("UNKNOWN", force=True)
        assert q.freshness == "unavailable"
        assert q.selection_trail[0]["outcome"] == "no_data"
        
    # Fifth query for AAPL must succeed and NOT be skipped by circuit breaker
    q_valid = svc.get_quote("AAPL", force=True)
    assert q_valid.freshness == "delayed"
    assert q_valid.price == 150.0
    assert q_valid.selection_trail[0]["outcome"] == "ok"
    assert prov.health.consecutive_failures == 0
    assert prov.health.no_data_queries == 4


def test_yahoo_quote_provider_falls_back_to_daily_when_1m_empty():
    """When market is closed and 1m history is empty, provider must fall back to
    5d daily history and return an end_of_day quote instead of raising no_data."""
    prov = md.YahooProvider()
    inst = md.resolve_instrument("AAPL")
    
    mock_ticker = MagicMock()
    mock_ticker.info = {}
    mock_ticker.fast_info = None
    mock_ticker.history.side_effect = [
        pd.DataFrame(),  # period='1d', interval='1m' is empty (closed market)
        pd.DataFrame({   # period='5d', interval='1d' returns daily bars
            "Open": [220.0, 222.0],
            "High": [223.0, 225.0],
            "Low": [219.0, 221.0],
            "Close": [221.5, 224.0],
            "Volume": [50000000, 48000000],
        }, index=pd.date_range("2026-10-15", periods=2, tz="America/New_York"))
    ]
    
    with patch("yfinance.Ticker", return_value=mock_ticker):
        raw = prov.fetch_raw(inst)
        assert raw["price"] == 224.0
        assert raw["prev_close"] == 221.5
        assert raw["data_type"] == "eod"
        assert raw["session"] == "closed"


def test_yahoo_quote_provider_hints_indian_suffix():
    """When a bare symbol fails with no_data, the error message must suggest exchange prefixes."""
    prov = md.YahooProvider()
    inst = md.resolve_instrument("TCS")  # Bare symbol resolved as US/UNSPECIFIED
    
    mock_ticker = MagicMock()
    mock_ticker.info = {}
    mock_ticker.fast_info = None
    mock_ticker.history.return_value = pd.DataFrame()  # Empty 1m and empty daily
    
    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(md.ProviderError) as exc_info:
            prov.fetch_raw(inst)
        msg = str(exc_info.value)
        assert "Yahoo returned no data for TCS." in msg
        assert "NSE:TCS" in msg or "TCS.NS" in msg
