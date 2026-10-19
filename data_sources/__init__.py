"""
data_sources
-------------
Unified entry point for stock data, covering effectively any stock on any
major exchange:

  - yfinance: the default, zero-setup source. Covers US exchanges (no
    suffix), most Indian stocks (.NS for NSE, .BO for BSE), and most other
    major world exchanges via their own suffix (.L London, .T Tokyo, .HK
    Hong Kong, .DE Germany, .PA Paris, .TO Toronto, .AX Australia, and more
    -- see data_sources/base.py's NON_INDIAN_SUFFIXES for the full list).
  - Upstox: Indian NSE/BSE equities via your own Upstox trading account --
    useful when yfinance is missing a ticker or rate-limited.
  - Groww: Indian NSE/BSE equities via your own Groww trading account -- an
    alternative to Upstox for the same purpose.

`fetch_stock_data(ticker, period, interval, source="auto")` is the one
function the rest of the project calls. In "auto" mode it tries yfinance
first (it needs no credentials and already covers the overwhelming majority
of tickers globally); if that fails, it falls through to Upstox/Groww ONLY
when the ticker is plausibly Indian (bare symbol or .NS/.BO) -- a ticker
with a suffix that clearly targets a different exchange (e.g. 'VOD.L') is
never sent to Upstox/Groww, since neither broker covers those markets and
trying would just waste a request and produce a confusing error.
"""

from . import yfinance_source, upstox_source, groww_source
from .base import DataSourceError, is_clearly_non_indian_ticker, INDIAN_SUFFIXES
import config


def _fetch_yfinance_with_suffix_fallback(ticker: str, period: str, interval: str, verbose: bool):
    """
    Tries the ticker exactly as given first. If that fails AND the ticker
    isn't already suffixed for a specific exchange (Indian or otherwise),
    retries the SAME symbol with '.NS' then '.BO' appended -- since
    yfinance itself already serves NSE/BSE data once the correct suffix is
    used, this resolves the common "typed a bare Indian symbol" case
    without needing any Upstox/Groww credentials at all.

    This only ever appends a recognized exchange suffix to the exact symbol
    the caller gave -- it never substitutes a different company/ticker.

    Returns (dataframe, errors_list) on success; raises DataSourceError with
    all attempts' messages if every variant fails.
    """
    errors = []
    try:
        return yfinance_source.fetch(ticker, period, interval), errors
    except DataSourceError as e:
        errors.append(f"yfinance ({ticker}): {e}")

    already_suffixed = ticker.upper().endswith(INDIAN_SUFFIXES) or is_clearly_non_indian_ticker(ticker)
    if not already_suffixed:
        for suffix in (".NS", ".BO"):
            candidate = f"{ticker}{suffix}"
            try:
                if verbose:
                    print(f"[data_sources] '{ticker}' not found as-is on yfinance -- "
                          f"retrying as '{candidate}'...")
                df = yfinance_source.fetch(candidate, period, interval)
                df.attrs["resolved_ticker"] = candidate      # visible: data is for `candidate`, not the symbol typed
                return df, errors
            except DataSourceError as e:
                errors.append(f"yfinance ({candidate}): {e}")

    raise DataSourceError("\n".join(errors))


def _exchange_kwargs(ticker: str) -> dict:
    """A '.BO' ticker is a BSE listing and must NEVER be served from NSE (or the reverse): the adapters default to
    NSE, so pass the exchange explicitly -- but only when it is BSE, which keeps every NSE/bare call unchanged."""
    return {"exchange": "BSE"} if ticker.strip().upper().endswith(".BO") else {}


def _fetch_raw(ticker: str, period: str = "2y", interval: str = "1d",
                      source: str = "auto", verbose: bool = True):
    """
    Returns a standardized OHLCV DataFrame (columns: Open, High, Low, Close,
    Volume; DatetimeIndex) for `ticker`, regardless of which underlying
    source ends up serving it.
    """
    source = source.lower()

    if source == "yfinance":
        df, _ = _fetch_yfinance_with_suffix_fallback(ticker, period, interval, verbose)
        return df
    if source == "upstox":
        return upstox_source.fetch(ticker, period, interval, **_exchange_kwargs(ticker))
    if source == "groww":
        return groww_source.fetch(ticker, period, interval, **_exchange_kwargs(ticker))
    if source != "auto":
        raise ValueError(f"Unknown source '{source}'. Use 'auto', 'yfinance', 'upstox', or 'groww'.")

    # --- auto mode: try each configured/available source in turn ---
    errors = []

    try:
        df, yf_errors = _fetch_yfinance_with_suffix_fallback(ticker, period, interval, verbose)
        return df
    except DataSourceError as e:
        errors.append(f"yfinance: {e}")
        if verbose:
            print(f"[data_sources] yfinance failed ({e}) -- trying fallbacks...")

    if is_clearly_non_indian_ticker(ticker):
        if verbose:
            print(f"[data_sources] '{ticker}' targets a non-Indian exchange -- "
                  f"skipping Upstox/Groww (they only cover NSE/BSE).")
        raise DataSourceError(
            "Could not fetch data for '" + ticker + "' from any source.\n" +
            "\n".join(errors) +
            "\n\nThis ticker's suffix targets a non-Indian exchange, so the "
            "Upstox/Groww fallbacks (Indian NSE/BSE only) were skipped. "
            "Double-check the symbol and suffix are correct for yfinance."
        )

    if config.UPSTOX_CLIENT_ID:
        try:
            if verbose:
                print("[data_sources] Trying Upstox...")
            return upstox_source.fetch(ticker, period, interval, **_exchange_kwargs(ticker))
        except DataSourceError as e:
            errors.append(f"Upstox: {e}")
            if verbose:
                print(f"[data_sources] Upstox failed: {e}")

    if config.GROWW_ACCESS_TOKEN or config.GROWW_API_KEY:
        try:
            if verbose:
                print("[data_sources] Trying Groww...")
            return groww_source.fetch(ticker, period, interval, **_exchange_kwargs(ticker))
        except DataSourceError as e:
            errors.append(f"Groww: {e}")
            if verbose:
                print(f"[data_sources] Groww failed: {e}")

    raise DataSourceError(
        "Could not fetch data for '" + ticker + "' from any source.\n" +
        "\n".join(errors) +
        "\n\nIf this is an Indian stock yfinance doesn't have, configure "
        "Upstox or Groww credentials in .env to enable those fallbacks."
    )



def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)          # indirection so tests can pin the clock


def fetch_stock_data(*args, **kwargs):
    """Fetch daily bars (see _fetch_raw) and drop a bar for a session that is still in progress. The classic /predict and
    CLI flows use this entry point; before V19 they treated an unfinished intraday bar as a settled close."""
    import config
    df = _fetch_raw(*args, **kwargs)
    if not getattr(config, "DROP_OPEN_SESSION_BAR", True):
        return df
    ticker = df.attrs.get("resolved_ticker") or (args[0] if args else kwargs.get("ticker"))
    try:
        import market_data, market_session
        inst = market_data.resolve_instrument(ticker)
    except Exception:  # noqa: BLE001 - an unparseable symbol must not break fetching
        return df
    out, dropped = market_session.drop_partial_bar(df, inst, None, now=_now())
    if dropped:
        out = out.copy()
        out.attrs = {**df.attrs, "dropped_open_session_bar": dropped}
    return out
