"""
data_sources/base.py
----------------------
Shared helpers used by every data source (yfinance, Upstox, Groww): date
range parsing and a common DataFrame shape every source must return, so the
rest of the pipeline (feature engineering, LSTM windowing, etc.) never needs
to know or care which source the data came from.
"""

import re
from datetime import date, timedelta

import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def period_to_daterange(period: str, end: date = None) -> tuple:
    """
    Converts a yfinance-style period string ('2y', '6mo', '30d', ...) into
    an explicit (from_date, to_date) pair of `date` objects. Upstox and
    Groww both want explicit dates rather than relative period strings.
    """
    end = end or date.today()
    match = re.match(r"^(\d+)(d|mo|y)$", period.strip().lower())
    if not match:
        raise ValueError(
            f"Could not parse period '{period}'. Expected formats like "
            f"'30d', '6mo', '2y'."
        )
    amount, unit = int(match.group(1)), match.group(2)
    if unit == "d":
        start = end - timedelta(days=amount)
    elif unit == "mo":
        start = end - timedelta(days=amount * 30)
    else:  # 'y'
        start = end - timedelta(days=amount * 365)
    return start, end


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures every source returns the same shape: a DatetimeIndex named
    implicitly by the index, and exactly the columns in REQUIRED_COLUMNS,
    sorted chronologically ascending, with no NaN rows.
    """
    df = df[REQUIRED_COLUMNS].copy()
    if getattr(df.index, "tz", None) is not None:
        # Yahoo can return a timezone-AWARE daily index. Everything downstream compares against naive dates, so a mix would
        # raise TypeError. Dropping the zone keeps the exchange-local wall-clock date (00:00 New York stays that trading date).
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna()
    return df


class DataSourceError(Exception):
    """Raised when a data source fails to return usable data (bad ticker,
    missing/expired credentials, API error, etc). Callers (the resolver)
    catch this to fall through to the next configured source."""
    pass


# ---------------------------------------------------------------------------
# Exchange-suffix awareness
# ---------------------------------------------------------------------------
# yfinance ticker suffixes for major non-US exchanges (not exhaustive, but
# covers the common ones). Used so the auto-fallback resolver knows Upstox/
# Groww (Indian-markets-only brokers) are only a sensible fallback for
# Indian tickers -- not, say, a London or Tokyo ticker that happens to also
# fail on yfinance for an unrelated reason.
INDIAN_SUFFIXES = (".NS", ".BO")
NON_INDIAN_SUFFIXES = (
    ".L",    # London Stock Exchange
    ".T",    # Tokyo Stock Exchange
    ".HK",   # Hong Kong Stock Exchange
    ".SS",   # Shanghai Stock Exchange
    ".SZ",   # Shenzhen Stock Exchange
    ".TO", ".V",   # Toronto / TSX Venture
    ".AX",   # Australian Securities Exchange
    ".DE", ".F",   # Germany (Xetra / Frankfurt)
    ".PA",   # Euronext Paris
    ".AS",   # Euronext Amsterdam
    ".MI",   # Borsa Italiana (Milan)
    ".SW",   # SIX Swiss Exchange
    ".MC",   # Bolsa de Madrid
    ".SI",   # Singapore Exchange
    ".KS", ".KQ",  # Korea Exchange
    ".SA",   # B3 (Brazil)
    ".NZ",   # New Zealand Exchange
    ".JO",   # Johannesburg Stock Exchange
)


def is_indian_ticker(ticker: str) -> bool:
    """True if the ticker explicitly targets NSE/BSE (.NS/.BO suffix)."""
    return ticker.upper().endswith(INDIAN_SUFFIXES)


def is_clearly_non_indian_ticker(ticker: str) -> bool:
    """
    True if the ticker's suffix clearly targets a non-Indian exchange (e.g.
    '.L' for London). A bare symbol with no suffix ('AAPL', 'RELIANCE') is
    NOT considered "clearly non-Indian" here -- it's ambiguous, since a bare
    symbol could plausibly be either, and Upstox/Groww's own instrument
    lookup will simply fail cleanly if it isn't one of theirs.
    """
    return ticker.upper().endswith(NON_INDIAN_SUFFIXES)
