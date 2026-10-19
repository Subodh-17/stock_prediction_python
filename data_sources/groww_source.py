"""
data_sources/groww_source.py
-------------------------------
Indian stock market data (NSE/BSE) via the Groww Trading API's historical
candle endpoint. An alternative to Upstox for the same purpose (Indian
equity coverage yfinance sometimes lacks or rate-limits).

AUTH MODEL -- two options, either works:

  Option A (simplest): generate an Access Token by hand from
  https://groww.in/user/profile/trading-apis ("Generate API keys" ->
  "Access Token"). Put it in .env as GROWW_ACCESS_TOKEN. It expires daily at
  6:00 AM, so you'll need to regenerate it once a day and update .env.

  Option B (automatable): generate a long-lived API Key + Secret from
  https://groww.in/trade-api/api-keys once, put them in .env as
  GROWW_API_KEY / GROWW_API_SECRET. This module then derives a fresh access
  token on demand via Groww's checksum-based "approval" flow (SHA-256 of
  secret + current epoch timestamp) -- no manual step needed *after* the key
  itself has been approved once on Groww's dashboard.

Groww's historical candle endpoint caps how much date range you can request
per call depending on candle interval (e.g. daily candles: 180 days per
request), so this module transparently chunks long requests and stitches
the results back together.
"""

import hashlib
import time
from datetime import date, timedelta

import pandas as pd
import requests

import config
import security
from .base import standardize_dataframe, DataSourceError

BASE_URL = "https://api.groww.in/v1"
TOKEN_URL = f"{BASE_URL}/token/api/access"
CANDLES_URL = f"{BASE_URL}/historical/candles"

# Groww's max days-per-request cap for daily candles.
MAX_DAYS_PER_REQUEST = 180


def _generate_checksum(secret: str, timestamp: str) -> str:
    return hashlib.sha256((secret + timestamp).encode("utf-8")).hexdigest()


def get_access_token() -> str:
    """
    Returns a usable access token: the static GROWW_ACCESS_TOKEN from .env
    if set (Option A), otherwise derives one from GROWW_API_KEY / _SECRET
    via the checksum "approval" flow (Option B).
    """
    if config.GROWW_ACCESS_TOKEN:
        return config.GROWW_ACCESS_TOKEN

    if not (config.GROWW_API_KEY and config.GROWW_API_SECRET):
        raise DataSourceError(
            "Groww requires either GROWW_ACCESS_TOKEN, or both GROWW_API_KEY "
            "and GROWW_API_SECRET, set in .env."
        )

    timestamp = str(int(time.time()))
    checksum = _generate_checksum(config.GROWW_API_SECRET, timestamp)
    resp = requests.post(
        TOKEN_URL,
        headers={"Authorization": f"Bearer {config.GROWW_API_KEY}", "Content-Type": "application/json"},
        json={"key_type": "approval", "checksum": checksum, "timestamp": timestamp},
        timeout=30,
    )
    if resp.status_code != 200:
        raise DataSourceError(
            "Groww token request failed: "
            f"{security.safe_upstream_detail(resp.text, resp.status_code)}"
        )

    token = resp.json().get("token")
    if not token:
        # VERIFIED LEAK, now closed: this is the response from Groww's TOKEN
        # endpoint -- the single most likely place in this project for a live
        # token to appear. A token returned under an unexpected key was
        # previously echoed verbatim into this exception message (and from
        # there into logs and the UI). The body is never quoted here now;
        # the status line alone is enough to diagnose a shape mismatch.
        raise DataSourceError(
            "Groww token response did not contain a 'token' field. The response "
            "body is deliberately not included here because it comes from an "
            "authentication endpoint. Check your GROWW_API_KEY/GROWW_API_SECRET "
            "and that the key has been approved on Groww's dashboard."
        )
    return token


def _fetch_chunk(groww_symbol: str, exchange: str, start_time: str, end_time: str,
                  access_token: str) -> list:
    resp = requests.get(
        CANDLES_URL,
        params={
            "exchange": exchange, "segment": "CASH", "groww_symbol": groww_symbol,
            "start_time": start_time, "end_time": end_time, "candle_interval": "1day",
        },
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
            "X-API-VERSION": "1.0",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise DataSourceError(
            "Groww historical candle request failed: "
            f"{security.safe_upstream_detail(resp.text, resp.status_code)}"
        )
    payload = resp.json()
    if payload.get("status") != "SUCCESS":
        raise DataSourceError(f"Groww API error: {payload.get('error')}")
    return payload.get("payload", {}).get("candles", [])


def fetch(ticker: str, period: str = "2y", interval: str = "1d", exchange: str = "NSE") -> pd.DataFrame:
    """
    The only function data_sources/__init__.py's resolver calls directly.
    Wrapped the same way as upstox_source.fetch(): the resolver's fallback
    logic only ever catches DataSourceError, so any unexpected failure in
    the token/chunking/request chain below is normalized to one here,
    rather than crashing the resolver outright.
    """
    if interval != "1d":
        raise DataSourceError("The Groww source in this project only implements daily candles ('1d').")

    try:
        from .base import period_to_daterange
        start_date, end_date = period_to_daterange(period)
        access_token = get_access_token()

        symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
        groww_symbol = f"{exchange}-{symbol}"

        all_candles = []
        chunk_start = start_date
        while chunk_start < end_date:
            chunk_end = min(chunk_start + timedelta(days=MAX_DAYS_PER_REQUEST), end_date)
            candles = _fetch_chunk(
                groww_symbol, exchange,
                f"{chunk_start.isoformat()} 09:15:00",
                f"{chunk_end.isoformat()} 15:30:00",
                access_token,
            )
            all_candles.extend(candles)
            chunk_start = chunk_end + timedelta(days=1)

        if not all_candles:
            raise DataSourceError(
                f"Groww returned no candles for '{groww_symbol}'. Double check the "
                f"trading symbol and exchange (NSE or BSE)."
            )

        rows = [{
            "Date": pd.to_datetime(c[0]),
            "Open": c[1], "High": c[2], "Low": c[3], "Close": c[4], "Volume": c[5],
        } for c in all_candles]
        df = pd.DataFrame(rows).set_index("Date")
        return standardize_dataframe(df)

    except DataSourceError:
        raise  # already the right type -- pass through unchanged
    except Exception as e:
        raise DataSourceError(
            f"Groww fetch for '{ticker}' failed unexpectedly: "
            f"{security.safe_exception_text(e)}"
        ) from e
