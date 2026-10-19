"""
data_sources/upstox_source.py
--------------------------------
Indian stock market data (NSE/BSE) via the Upstox API v3 Historical Candle
endpoint. Useful when yfinance's Indian coverage is missing a ticker, rate
limited, or you specifically want data through your own Upstox account.

AUTH MODEL (this is how Upstox actually works, not a simplification):
Upstox uses OAuth2 authorization-code login -- there's no way to fetch data
with just a static API key, by design (it's tied to your real trading
account for security/compliance). So the flow here is:

  1. First run of the day: this module prints a login URL, you open it in a
     browser, log in, and paste back either the full redirect URL or just
     the `code` query param.
  2. That code is exchanged for an access_token, which is cached to
     `.upstox_token_cache.json` and reused for the rest of the day (Upstox
     tokens expire daily around 3:30 AM IST, matching their cache TTL here).
  3. Every subsequent call this same day reuses the cached token with no
     further prompts.

Requires in your .env: UPSTOX_CLIENT_ID, UPSTOX_CLIENT_SECRET, UPSTOX_REDIRECT_URI
(all three come from your app at https://account.upstox.com/developer/apps)
"""

import gzip
import json
import os
from datetime import date, datetime
from urllib.parse import urlencode, urlparse, parse_qs

import pandas as pd
import requests

import config
import security
from .base import standardize_dataframe, period_to_daterange, DataSourceError

TOKEN_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".upstox_token_cache.json")
INSTRUMENT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".upstox_instruments_cache.json")
LOGIN_DIALOG_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
INSTRUMENTS_URL = {
    "NSE": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    "BSE": "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz",
}
CANDLE_URL_TEMPLATE = "https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"


def _load_cached_token() -> str:
    if not os.path.exists(TOKEN_CACHE_PATH):
        return None
    try:
        with open(TOKEN_CACHE_PATH) as f:
            cache = json.load(f)
        if cache.get("date") == date.today().isoformat():
            return cache.get("access_token")
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_token_cache(access_token: str):
    with open(TOKEN_CACHE_PATH, "w") as f:
        json.dump({"date": date.today().isoformat(), "access_token": access_token}, f)


def _interactive_login() -> str:
    """Walks the user through the OAuth login flow in the terminal."""
    if not (config.UPSTOX_CLIENT_ID and config.UPSTOX_CLIENT_SECRET and config.UPSTOX_REDIRECT_URI):
        raise DataSourceError(
            "Upstox requires UPSTOX_CLIENT_ID, UPSTOX_CLIENT_SECRET, and "
            "UPSTOX_REDIRECT_URI to be set in .env (from your app at "
            "https://account.upstox.com/developer/apps)."
        )

    params = {
        "client_id": config.UPSTOX_CLIENT_ID,
        "redirect_uri": config.UPSTOX_REDIRECT_URI,
        "response_type": "code",
    }
    auth_url = f"{LOGIN_DIALOG_URL}?{urlencode(params)}"

    print("\n--- Upstox login required (once per day) ---")
    print(f"1. Open this URL in a browser and log in:\n   {auth_url}")
    print("2. After login you'll be redirected to your redirect_uri with a ?code=... param.")
    pasted = input("3. Paste the FULL redirect URL (or just the code) here: ").strip()

    if pasted.startswith("http"):
        code = parse_qs(urlparse(pasted).query).get("code", [None])[0]
    else:
        code = pasted
    if not code:
        raise DataSourceError("Could not find an authorization code in what you pasted.")

    resp = requests.post(
        TOKEN_URL,
        data={
            "code": code,
            "client_id": config.UPSTOX_CLIENT_ID,
            "client_secret": config.UPSTOX_CLIENT_SECRET,
            "redirect_uri": config.UPSTOX_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise DataSourceError(
            "Upstox token exchange failed: "
            f"{security.safe_upstream_detail(resp.text, resp.status_code)}"
        )

    access_token = resp.json().get("access_token")
    if not access_token:
        # Same verified leak class as the Groww token endpoint: this body is
        # an OAuth token-exchange response. Never quoted.
        raise DataSourceError(
            "Upstox token response did not contain an access_token. The response "
            "body is deliberately not included here because it comes from an "
            "authentication endpoint. Re-check UPSTOX_CLIENT_ID / "
            "UPSTOX_CLIENT_SECRET / UPSTOX_REDIRECT_URI and try logging in again."
        )

    _save_token_cache(access_token)
    return access_token


def get_access_token() -> str:
    token = _load_cached_token()
    if token:
        return token
    return _interactive_login()


def _load_instrument_map(exchange: str = "NSE") -> dict:
    """
    Downloads (and caches to disk) the Upstox instrument master for the given
    exchange, returning a {trading_symbol: instrument_key} map filtered to
    plain equities (instrument_type == 'EQ').
    """
    cache_file = INSTRUMENT_CACHE_PATH.replace(".json", f"_{exchange}.json")
    if os.path.exists(cache_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_file)).date()
        if mtime == date.today():
            with open(cache_file) as f:
                return json.load(f)

    url = INSTRUMENTS_URL.get(exchange)
    if url is None:
        raise DataSourceError(f"No instrument file known for exchange '{exchange}'")

    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise DataSourceError(f"Failed to download Upstox instrument file: {resp.status_code}")

    raw = gzip.decompress(resp.content)
    instruments = json.loads(raw)

    symbol_map = {
        inst["trading_symbol"]: inst["instrument_key"]
        for inst in instruments
        if inst.get("instrument_type") == "EQ" and inst.get("trading_symbol")
    }
    with open(cache_file, "w") as f:
        json.dump(symbol_map, f)
    return symbol_map


def _resolve_instrument_key(ticker: str, exchange: str = "NSE") -> str:
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    symbol_map = _load_instrument_map(exchange)
    if symbol not in symbol_map:
        raise DataSourceError(
            f"'{symbol}' not found in the Upstox {exchange} instrument list. "
            f"Double check the trading symbol (e.g. 'RELIANCE', 'TCS', 'INFY')."
        )
    return symbol_map[symbol]


def _parse_candles(candles: list) -> pd.DataFrame:
    # Each candle: [timestamp, open, high, low, close, volume, open_interest]
    rows = []
    for c in candles:
        rows.append({
            "Date": pd.to_datetime(c[0]),
            "Open": c[1], "High": c[2], "Low": c[3], "Close": c[4], "Volume": c[5],
        })
    df = pd.DataFrame(rows).set_index("Date")
    return df


def fetch(ticker: str, period: str = "2y", interval: str = "1d", exchange: str = "NSE") -> pd.DataFrame:
    """
    The only function data_sources/__init__.py's resolver calls directly.
    Everything below this point -- the OAuth login flow, instrument lookup,
    and the candle request itself -- can fail in ways beyond the explicit
    status-code checks already in place (a dropped connection, a malformed
    JSON body, etc.). The resolver's auto-fallback logic only ever catches
    DataSourceError, so ANY failure in this call chain is normalized to one
    here -- otherwise a raw network exception would crash the whole resolver
    instead of letting it fall through to Groww or yfinance.
    """
    if interval != "1d":
        raise DataSourceError("The Upstox source in this project only implements daily candles ('1d').")

    try:
        access_token = get_access_token()
        instrument_key = _resolve_instrument_key(ticker, exchange)
        from_date, to_date = period_to_daterange(period)

        url = CANDLE_URL_TEMPLATE.format(
            instrument_key=instrument_key, unit="days", interval="1",
            to_date=to_date.isoformat(), from_date=from_date.isoformat(),
        )
        resp = requests.get(url, headers={
            "Authorization": f"Bearer {access_token}", "Accept": "application/json",
        }, timeout=30)
        if resp.status_code != 200:
            raise DataSourceError(
            "Upstox historical candle request failed: "
            f"{security.safe_upstream_detail(resp.text, resp.status_code)}"
        )

        payload = resp.json()
        candles = payload.get("data", {}).get("candles", [])
        if not candles:
            raise DataSourceError(f"Upstox returned no candles for '{ticker}' in this date range.")

        df = _parse_candles(candles)
        return standardize_dataframe(df)

    except DataSourceError:
        raise  # already the right type -- pass through unchanged
    except Exception as e:
        raise DataSourceError(
            f"Upstox fetch for '{ticker}' failed unexpectedly: "
            f"{security.safe_exception_text(e)}"
        ) from e
