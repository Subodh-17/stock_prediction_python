"""
data_sources/yfinance_source.py
----------------------------------
Global stock coverage (US, and most international exchanges including NSE/BSE
via .NS / .BO ticker suffixes) via Yahoo Finance. No API key required -- this
is the default, zero-setup source.

Reliability notes:
  - A short retry loop with exponential backoff handles transient network
    hiccups and Yahoo Finance rate-limiting (HTTP 429 or connection resets),
    which are common enough with yfinance's unofficial API to be worth
    absorbing automatically rather than failing a whole pipeline run.
  - A genuinely EMPTY result (ticker doesn't exist / delisted) is NOT
    retried -- that's a real answer, not a transient failure, and retrying
    it would just waste time before producing the same correct error.
  - An explicit request timeout is set (config.DATA_FETCH_TIMEOUT_SECONDS)
    so a hung connection can't block the pipeline indefinitely.
"""

import time

import pandas as pd

import config
from .base import standardize_dataframe, DataSourceError

# Substrings that indicate a rate-limit / transient-server response worth
# retrying, as seen in yfinance's underlying urllib3/requests exceptions
# (which don't expose a clean status-code attribute in all versions).
_RETRYABLE_HINTS = ("429", "too many requests", "rate limit", "timed out",
                     "timeout", "connection", "temporarily unavailable", "503", "502")


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _RETRYABLE_HINTS)


def fetch(ticker: str, period: str = "2y", interval: str = "1d",
          max_retries: int = None, timeout: float = None) -> pd.DataFrame:
    import yfinance as yf

    max_retries = config.DATA_FETCH_MAX_RETRIES if max_retries is None else max_retries
    timeout = config.DATA_FETCH_TIMEOUT_SECONDS if timeout is None else timeout

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                              progress=False, auto_adjust=True, timeout=timeout)

            if df is None or df.empty:
                # Not a transient problem -- don't retry, just report clearly.
                raise DataSourceError(
                    f"yfinance returned no data for '{ticker}'. Check the symbol "
                    f"(Indian stocks need a .NS or .BO suffix, e.g. 'RELIANCE.NS')."
                )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return standardize_dataframe(df)

        except DataSourceError:
            raise  # empty-data case above -- already the right type, don't retry

        except Exception as e:
            last_error = e
            if attempt < max_retries and _is_retryable(e):
                wait = config.DATA_FETCH_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(wait)
                continue
            break

    raise DataSourceError(
        f"yfinance request for '{ticker}' failed after "
        f"{max_retries + 1} attempt(s): {last_error}"
    ) from last_error
