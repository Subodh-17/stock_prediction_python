"""
Unit tests for data_sources/ (Upstox, Groww, and the auto-fallback
resolver), using mocked HTTP responses since this sandbox can't reach
api.upstox.com / api.groww.in directly. Validates the business logic
(date-range math, candle parsing, chunking, checksum generation, fallback
ordering) that real credentials + network access would otherwise be needed
to exercise end to end.
"""
import os
import sys
from datetime import date
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_sources.base import period_to_daterange, standardize_dataframe, DataSourceError, \
    is_indian_ticker, is_clearly_non_indian_ticker
from data_sources import upstox_source, groww_source
import data_sources
import pandas as pd

# ---------------------------------------------------------------------------
# base.py
# ---------------------------------------------------------------------------
d1, d2 = period_to_daterange("2y", end=date(2026, 7, 22))
assert d2 == date(2026, 7, 22)
assert d1 == date(2024, 7, 23) or (d2 - d1).days in (730, 731), (d1, d2)
print("Test 1 (period_to_daterange '2y') passed:", d1, "->", d2)

d1, d2 = period_to_daterange("6mo", end=date(2026, 7, 22))
assert (d2 - d1).days == 180, (d1, d2)
print("Test 2 (period_to_daterange '6mo') passed:", d1, "->", d2)

try:
    period_to_daterange("banana")
    raise AssertionError("Should have raised on invalid period string")
except ValueError:
    print("Test 3 (period_to_daterange invalid input -> ValueError) passed")

# ---------------------------------------------------------------------------
# upstox_source: candle parsing
# ---------------------------------------------------------------------------
mock_candles = [
    ["2026-07-20T00:00:00+05:30", 100.0, 105.0, 99.0, 103.0, 50000, None],
    ["2026-07-21T00:00:00+05:30", 103.0, 106.0, 102.0, 104.5, 48000, None],
]
df = upstox_source._parse_candles(mock_candles)
assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
assert df.shape[0] == 2
assert df["Close"].iloc[-1] == 104.5
print("Test 4 (upstox candle parsing) passed:\n", df)

# ---------------------------------------------------------------------------
# upstox_source: instrument resolution with a mocked instrument file download
# ---------------------------------------------------------------------------
import gzip, json as jsonlib

fake_instruments = [
    {"trading_symbol": "RELIANCE", "instrument_key": "NSE_EQ|INE002A01018", "instrument_type": "EQ"},
    {"trading_symbol": "TCS", "instrument_key": "NSE_EQ|INE467B01029", "instrument_type": "EQ"},
    {"trading_symbol": "NIFTY", "instrument_key": "NSE_INDEX|Nifty 50", "instrument_type": "INDEX"},
]
fake_gz = gzip.compress(jsonlib.dumps(fake_instruments).encode())

with patch("data_sources.upstox_source.requests.get") as mock_get, \
     patch("os.path.exists", return_value=False):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = fake_gz
    mock_get.return_value = mock_resp
    key = upstox_source._resolve_instrument_key("RELIANCE", "NSE")
    assert key == "NSE_EQ|INE002A01018", key
    print("Test 5 (upstox instrument resolution, mocked download) passed:", key)

    try:
        upstox_source._resolve_instrument_key("NOTASTOCK", "NSE")
        raise AssertionError("Should have raised DataSourceError for unknown symbol")
    except DataSourceError:
        print("Test 6 (upstox unknown symbol -> DataSourceError) passed")

# ---------------------------------------------------------------------------
# groww_source: checksum generation matches the documented algorithm exactly
# ---------------------------------------------------------------------------
import hashlib
secret, ts = "my-secret", "1719830400"
expected = hashlib.sha256((secret + ts).encode("utf-8")).hexdigest()
actual = groww_source._generate_checksum(secret, ts)
assert actual == expected, (actual, expected)
print("Test 7 (groww checksum generation) passed:", actual)

# ---------------------------------------------------------------------------
# groww_source: chunking logic for date ranges longer than MAX_DAYS_PER_REQUEST
# ---------------------------------------------------------------------------
call_log = []

def fake_fetch_chunk(groww_symbol, exchange, start_time, end_time, access_token):
    call_log.append((start_time, end_time))
    # Return one fake candle per chunk so we can confirm stitching works
    day = start_time.split(" ")[0]
    return [[f"{day}T09:15:00", 100.0, 101.0, 99.0, 100.5, 1000, None]]

with patch.object(groww_source, "_fetch_chunk", side_effect=fake_fetch_chunk), \
     patch.object(groww_source, "get_access_token", return_value="fake-token"):
    result_df = groww_source.fetch("RELIANCE", period="1y")
    n_chunks_expected = -(-365 // groww_source.MAX_DAYS_PER_REQUEST)  # ceil division
    assert len(call_log) == n_chunks_expected, (len(call_log), n_chunks_expected)
    assert result_df.shape[0] == len(call_log)
    print(f"Test 8 (groww chunking: 1y split into {len(call_log)} requests of "
          f"<= {groww_source.MAX_DAYS_PER_REQUEST} days) passed")

# ---------------------------------------------------------------------------
# resolver: auto mode falls through from yfinance -> upstox on failure
# ---------------------------------------------------------------------------
fake_ohlcv = pd.DataFrame({
    "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1000],
}, index=pd.to_datetime(["2026-07-21"]))

with patch.object(data_sources.yfinance_source, "fetch", side_effect=DataSourceError("no data")), \
     patch.object(data_sources.upstox_source, "fetch", return_value=fake_ohlcv), \
     patch.object(data_sources.config, "UPSTOX_CLIENT_ID", "fake_id"), \
     patch.object(data_sources.config, "GROWW_ACCESS_TOKEN", ""), \
     patch.object(data_sources.config, "GROWW_API_KEY", ""):
    result = data_sources.fetch_stock_data("RELIANCE.NS", source="auto", verbose=False)
    assert result.equals(fake_ohlcv)
    print("Test 9 (resolver: yfinance fails -> falls back to Upstox) passed")

# ---------------------------------------------------------------------------
# resolver: auto mode raises a clear combined error when everything fails
# and no fallback credentials are configured
# ---------------------------------------------------------------------------
with patch.object(data_sources.yfinance_source, "fetch", side_effect=DataSourceError("no data")), \
     patch.object(data_sources.config, "UPSTOX_CLIENT_ID", ""), \
     patch.object(data_sources.config, "GROWW_ACCESS_TOKEN", ""), \
     patch.object(data_sources.config, "GROWW_API_KEY", ""):
    try:
        data_sources.fetch_stock_data("FAKETICKER", source="auto", verbose=False)
        raise AssertionError("Should have raised DataSourceError")
    except DataSourceError as e:
        assert "FAKETICKER" in str(e)
        print("Test 10 (resolver: all sources fail -> clear combined error) passed")

# ---------------------------------------------------------------------------
# exchange-suffix awareness
# ---------------------------------------------------------------------------
assert is_indian_ticker("RELIANCE.NS")
assert is_indian_ticker("tcs.bo")  # case-insensitive
assert not is_indian_ticker("AAPL")
print("Test 11 (is_indian_ticker) passed")

assert is_clearly_non_indian_ticker("VOD.L")
assert is_clearly_non_indian_ticker("7203.T")
assert not is_clearly_non_indian_ticker("AAPL")       # bare symbol -- ambiguous, not "clearly non-Indian"
assert not is_clearly_non_indian_ticker("RELIANCE.NS")
print("Test 12 (is_clearly_non_indian_ticker) passed")

# resolver: a clearly non-Indian ticker must NEVER be sent to Upstox/Groww,
# even if both are fully configured
with patch.object(data_sources.yfinance_source, "fetch", side_effect=DataSourceError("no data")), \
     patch.object(data_sources.upstox_source, "fetch") as mock_upstox, \
     patch.object(data_sources.groww_source, "fetch") as mock_groww, \
     patch.object(data_sources.config, "UPSTOX_CLIENT_ID", "fake_id"), \
     patch.object(data_sources.config, "GROWW_ACCESS_TOKEN", "fake_token"):
    try:
        data_sources.fetch_stock_data("VOD.L", source="auto", verbose=False)
        raise AssertionError("Should have raised DataSourceError")
    except DataSourceError as e:
        mock_upstox.assert_not_called()
        mock_groww.assert_not_called()
        assert "non-Indian" in str(e)
        print("Test 13 (resolver: non-Indian ticker never reaches Upstox/Groww) passed")

# resolver: a bare/ambiguous symbol (no suffix) still gets the fallback treatment
with patch.object(data_sources.yfinance_source, "fetch", side_effect=DataSourceError("no data")), \
     patch.object(data_sources.upstox_source, "fetch", return_value=fake_ohlcv) as mock_upstox, \
     patch.object(data_sources.config, "UPSTOX_CLIENT_ID", "fake_id"), \
     patch.object(data_sources.config, "GROWW_ACCESS_TOKEN", ""), \
     patch.object(data_sources.config, "GROWW_API_KEY", ""):
    result = data_sources.fetch_stock_data("RELIANCE", source="auto", verbose=False)
    mock_upstox.assert_called_once()
    print("Test 14 (resolver: bare/ambiguous symbol still falls back to Upstox) passed")

# ---------------------------------------------------------------------------
# regression test: a RAW exception (not DataSourceError) from deep inside
# Upstox's call chain -- e.g. a network-level error mid-request -- must
# still be caught and normalized, not crash the whole resolver. Previously
# only explicit DataSourceError was guaranteed; an uncaught requests
# exception from get_access_token()/_resolve_instrument_key() would have
# propagated straight through fetch() and broken the auto-fallback chain.
# ---------------------------------------------------------------------------
with patch.object(upstox_source, "get_access_token", side_effect=ConnectionError("connection reset")):
    try:
        upstox_source.fetch("RELIANCE", period="1y")
        raise AssertionError("Should have raised DataSourceError")
    except DataSourceError as e:
        assert "unexpectedly" in str(e).lower() or "connection reset" in str(e).lower()
        print("Test 15 (upstox_source.fetch: raw ConnectionError normalized to DataSourceError) passed")
    except ConnectionError:
        raise AssertionError("BUG: raw ConnectionError leaked out of upstox_source.fetch() uncaught")

with patch.object(groww_source, "get_access_token", side_effect=TimeoutError("timed out")):
    try:
        groww_source.fetch("TCS", period="1y")
        raise AssertionError("Should have raised DataSourceError")
    except DataSourceError as e:
        print("Test 16 (groww_source.fetch: raw TimeoutError normalized to DataSourceError) passed")
    except TimeoutError:
        raise AssertionError("BUG: raw TimeoutError leaked out of groww_source.fetch() uncaught")

# and the resolver itself must still fall through cleanly when the first
# source raises something other than DataSourceError internally
with patch.object(data_sources.yfinance_source, "fetch", side_effect=DataSourceError("no data")), \
     patch.object(upstox_source, "get_access_token", side_effect=ConnectionError("boom")), \
     patch.object(data_sources.groww_source, "fetch", return_value=fake_ohlcv) as mock_groww, \
     patch.object(data_sources.config, "UPSTOX_CLIENT_ID", "fake_id"), \
     patch.object(data_sources.config, "GROWW_ACCESS_TOKEN", "fake_token"):
    result = data_sources.fetch_stock_data("RELIANCE", source="auto", verbose=False)
    mock_groww.assert_called_once()
    print("Test 17 (resolver: survives a raw exception from Upstox internals, still reaches Groww) passed")

print("\nALL DATA SOURCE TESTS PASSED")
