"""
data_loader.py
---------------
Fetches historical OHLCV stock data (via yfinance) and prepares two parallel
representations of it:

  1. A flat feature table for Linear Regression (lagged closing prices +
     simple technical features -> next day's close).
  2. A windowed 3D tensor for the LSTM (sequences of `lookback` days of
     scaled closing prices -> next day's scaled close).

Both share the same train/test split boundary (chronological, not random --
shuffling time series data leaks future information into training).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List
from sklearn.preprocessing import MinMaxScaler

import config
import perf
from data_sources import fetch_stock_data as _fetch_from_sources


class InsufficientDataError(Exception):
    """Raised when there isn't enough historical data for a meaningful
    train/test split -- e.g. a short --period combined with the default
    LOOKBACK_WINDOW. Raised early with an actionable message instead of
    letting numpy/sklearn fail later with a confusing shape/index error."""
    pass


def fetch_stock_data(ticker: str = None, period: str = None, interval: str = None,
                      source: str = None) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for `ticker`, covering effectively any
    stock: global/US tickers and most Indian tickers (.NS/.BO) via yfinance
    by default, with automatic fallback to Upstox or Groww (if configured in
    .env) for Indian equities yfinance can't serve. See data_sources/ for
    the full multi-source logic.
    """
    ticker = ticker or config.TICKER
    period = period or config.PERIOD
    interval = interval or config.INTERVAL
    source = source or config.DATA_SOURCE
    return _fetch_from_sources(ticker, period, interval, source=source)


def validate_sufficient_data(df: pd.DataFrame, lookback: int = None, min_test_rows: int = 10):
    """
    Confirms there's enough data for both the LSTM's lookback window and a
    non-trivial train/test split on both the LR and LSTM sides. Call this
    right after fetching data, before any feature engineering -- it's much
    easier to act on a clear message here than on a shape-mismatch error
    three functions deep.
    """
    lookback = lookback or config.LOOKBACK_WINDOW
    # After windowing, usable LSTM rows = len(df) - lookback. We need enough
    # of those left over for both training and a real test set.
    min_required = lookback + min_test_rows + 10  # +10 buffer for feature-engineering dropna
    if len(df) < min_required:
        raise InsufficientDataError(
            f"Only {len(df)} rows of data available, but at least {min_required} are "
            f"needed (LOOKBACK_WINDOW={lookback} + a buffer for a usable train/test "
            f"split). Use a longer --period (e.g. '6mo' or '1y'), or lower "
            f"LOOKBACK_WINDOW in .env if you specifically need to work with less history."
        )


@dataclass
class DataQualityReport:
    ticker: str
    n_rows: int
    issues: List[str] = field(default_factory=list)     # problems serious enough to distrust the dataset
    warnings: List[str] = field(default_factory=list)    # notable but not necessarily wrong
    is_valid: bool = True

    def to_dict(self) -> dict:
        return {"ticker": self.ticker, "n_rows": self.n_rows, "issues": self.issues,
                "warnings": self.warnings, "is_valid": self.is_valid}

    def summary_text(self) -> str:
        if not self.issues and not self.warnings:
            return f"Data quality: OK ({self.n_rows} rows, no issues found)."
        lines = [f"Data quality check for {self.ticker or 'dataset'} ({self.n_rows} rows):"]
        for i in self.issues:
            lines.append(f"  ISSUE:   {i}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def check_data_quality(df: pd.DataFrame, ticker: str = "") -> DataQualityReport:
    """
    Diagnostic-only data-quality check on fetched OHLCV data. This NEVER
    modifies `df` -- silently "fixing" (dropping duplicate rows, filling
    gaps, clipping outliers, etc.) could quietly change what the user
    thinks they're analyzing, so this only reports what it finds and lets
    the caller decide what to do about it (the pipeline currently proceeds
    with a logged warning rather than blocking on anything short of
    completely empty/missing data, which validate_sufficient_data already
    catches separately).

    Checks: missing (NaN) values, duplicate dates, chronological ordering,
    negative/zero prices, negative volume, invalid OHLC relationships
    (High below the other prices, or Low above them), unusually large
    single-day price moves, and unusually large gaps between consecutive
    rows. The price/volume/OHLC checks are reported as ISSUES (the data is
    probably wrong); large moves/gaps are reported as WARNINGS (worth a
    second look, but real markets do sometimes do this).
    """
    issues: List[str] = []
    warnings: List[str] = []

    if df is None or len(df) == 0:
        issues.append("No data returned (0 rows).")
        return DataQualityReport(ticker=ticker, n_rows=0, issues=issues, warnings=warnings, is_valid=False)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required column(s): {missing_cols}")
        return DataQualityReport(ticker=ticker, n_rows=len(df), issues=issues,
                                  warnings=warnings, is_valid=False)

    na_counts = df[required_cols].isna().sum()
    total_na = int(na_counts.sum())
    if total_na > 0:
        detail = {c: int(v) for c, v in na_counts.items() if v > 0}
        issues.append(f"{total_na} missing (NaN) value(s) across OHLCV columns: {detail}")

    n_inf = int(np.isinf(df[required_cols].to_numpy(dtype=float, na_value=np.nan)).sum())
    if n_inf > 0:
        issues.append(f"{n_inf} non-finite (infinite) value(s) in OHLCV columns.")

    if getattr(df.index, "tz", None) is not None:
        warnings.append(f"The index is timezone-aware ({df.index.tz}); the data sources normalise it to exchange-local "
                        f"dates, but a raw frame like this one would not compare cleanly with naive dates.")

    n_dupes = int(df.index.duplicated().sum())
    if n_dupes > 0:
        issues.append(f"{n_dupes} duplicate date(s) found in the index.")

    if not df.index.is_monotonic_increasing:
        issues.append("Dates are not in chronological (increasing) order.")

    price_cols = ["Open", "High", "Low", "Close"]
    non_positive = (df[price_cols] <= 0).any(axis=1)
    if non_positive.any():
        issues.append(f"{int(non_positive.sum())} row(s) have a zero or negative Open/High/Low/Close.")

    if (df["Volume"] < 0).any():
        issues.append(f"{int((df['Volume'] < 0).sum())} row(s) have negative Volume.")

    # OHLC relationship: High must be the max of the day's prices, Low the min.
    invalid_high = df["High"] < df[["Open", "Close", "Low"]].max(axis=1)
    invalid_low = df["Low"] > df[["Open", "Close", "High"]].min(axis=1)
    n_invalid_ohlc = int((invalid_high | invalid_low).sum())
    if n_invalid_ohlc > 0:
        issues.append(f"{n_invalid_ohlc} row(s) have an impossible OHLC relationship "
                       f"(High below another price, or Low above another price).")

    if len(df) > 1:
        pct_changes = df["Close"].pct_change().abs()
        extreme_moves = pct_changes[pct_changes > 0.5]  # >50% single-day move
        if len(extreme_moves) > 0:
            warnings.append(
                f"{len(extreme_moves)} day(s) with a >50% single-day Close move -- could be a "
                f"genuine crash/rally, an unadjusted stock split, or bad data. Not treated as an "
                f"error on its own since real markets occasionally do move this much."
            )

        try:
            gaps_days = df.index.to_series().diff().dt.days.dropna()
            large_gaps = gaps_days[gaps_days > 10]
            if len(large_gaps) > 0:
                warnings.append(
                    f"{len(large_gaps)} gap(s) of more than 10 calendar days between consecutive "
                    f"rows -- could indicate missing data or an extended market closure."
                )
        except (AttributeError, TypeError):
            pass  # index isn't datetime-like; gap check doesn't apply

    # Frozen / stale feed: many consecutive sessions where Open == High == Low == Close. A real, liquid market almost never
    # prints this; a provider repeating its last value does. Reported as a WARNING (an illiquid instrument can do it too).
    def _longest_run(mask) -> int:
        best = run = 0
        for v in np.asarray(mask, dtype=bool):
            run = run + 1 if v else 0
            best = max(best, run)
        return best

    try:
        flat = ((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])).to_numpy()
        run_flat = _longest_run(flat)
        if run_flat >= 5:
            warnings.append(f"{run_flat} consecutive day(s) with identical Open/High/Low/Close -- possible stale or frozen "
                            f"data feed (or a very illiquid instrument). Results for this window may not reflect real trading.")
        vol = df["Volume"].to_numpy(dtype=float, na_value=np.nan)
        if np.nanmax(vol, initial=0) > 0:                       # instruments that never report volume (indices) are exempt
            run_zero = _longest_run(vol == 0)
            if run_zero >= 5:
                warnings.append(f"{run_zero} consecutive day(s) with zero Volume on an instrument that otherwise reports "
                                f"volume -- possible missing volume data or a trading halt.")
    except (TypeError, ValueError):
        pass

    return DataQualityReport(ticker=ticker, n_rows=len(df), issues=issues,
                              warnings=warnings, is_valid=(len(issues) == 0))


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds simple engineered features used by the linear regression model."""
    out = df.copy()
    out["Return_1d"] = out["Close"].pct_change(1)
    out["MA_5"] = out["Close"].rolling(5).mean()
    out["MA_10"] = out["Close"].rolling(10).mean()
    out["Volatility_5"] = out["Close"].rolling(5).std()
    out["Lag_1"] = out["Close"].shift(1)
    out["Lag_2"] = out["Close"].shift(2)
    out["Lag_3"] = out["Close"].shift(3)
    out["Target"] = out["Close"].shift(-1)  # predict next day's close
    out = out.dropna()
    return out


LR_FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume",
                       "Return_1d", "MA_5", "MA_10", "Volatility_5",
                       "Lag_1", "Lag_2", "Lag_3"]


def lr_target_dates(df: pd.DataFrame, featured_index) -> pd.DatetimeIndex:
    """
    Maps each engineered feature row to the date its `Target` value actually
    belongs to -- i.e. the NEXT trading day after the feature row's own date,
    since add_technical_features() sets Target = Close.shift(-1).

    WHY THIS EXISTS (a real bug it fixes): add_technical_features() labels
    each row with the date the FEATURES were observed, but the label (y) on
    that row is the following day's close. Returning the feature date as
    "the date of this test point" therefore mislabels every LR test
    observation by exactly one trading day. That mislabeling is invisible
    inside the LR model itself (features and labels stay correctly paired),
    but it matters everywhere the LR series is lined up against something
    else that IS labeled by realization date:

      - the LR+LSTM ensemble backtest joins the two models on these date
        keys, so it was blending an LR forecast of close(d+1) with an LSTM
        forecast of close(d) and scoring the result against close(d+1);
      - the CSV/PDF export table put an `actual` of close(d+1) on the same
        row as an `lstm_predicted` for close(d);
      - the actual-vs-predicted chart drew the LR lines one trading day to
        the left of the LSTM line;
      - walk-forward fold boundaries were reported one day early, which made
        backtest._compute_historical_dynamic_weights' "only folds that
        concluded strictly BEFORE this day" filter one day too permissive.

    The LSTM side was already labeled by realization date (see
    prepare_lstm_return_data), so aligning LR to the same convention makes
    the two directly comparable rather than silently offset.

    `featured_index` must be an index produced by add_technical_features(df)
    on this same `df` -- every entry must exist in df.index and must not be
    df's final row (which add_technical_features drops precisely because its
    Target is unknown).
    """
    positions = df.index.get_indexer(pd.Index(featured_index))
    if (positions < 0).any():
        raise ValueError(
            "featured_index contains dates that are not present in df.index -- "
            "lr_target_dates() must be called with the index produced by "
            "add_technical_features() on this same DataFrame."
        )
    if (positions + 1 >= len(df.index)).any():
        raise ValueError(
            "featured_index includes df's final row, which has no next-day Target. "
            "add_technical_features() drops that row; something has re-added it."
        )
    return df.index[positions + 1]


def prepare_linear_regression_data(df: pd.DataFrame, split_ratio: float = None):
    """
    Returns X_train, X_test, y_train, y_test, feature_names, dates_test.

    `dates_test` is labeled by TARGET date (the day each y_test value was
    actually realized), not by the date the features were observed -- see
    lr_target_dates() for why that distinction is load-bearing.
    """
    split_ratio = split_ratio or config.TRAIN_TEST_SPLIT
    featured = add_technical_features(df)
    feature_cols = LR_FEATURE_COLUMNS

    X = featured[feature_cols].values
    y = featured["Target"].values
    dates = lr_target_dates(df, featured.index)

    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    return X_train, X_test, y_train, y_test, feature_cols, dates_test


def build_latest_feature_row(df: pd.DataFrame) -> np.ndarray:
    """
    Builds a single feature row (in LR_FEATURE_COLUMNS order) representing
    the MOST RECENT day in `df` -- i.e. today -- with every lookback-based
    feature (moving averages, lag prices, volatility, 1-day return) computed
    using only historical data up through and including today. Feeding this
    into a trained model's predict_next() gives a genuine forecast for a day
    that isn't anywhere in `df` yet (tomorrow).

    WHY THIS FUNCTION EXISTS (a real bug it fixes): add_technical_features()
    sets Target = Close.shift(-1) and then drops rows with a NaN Target --
    which is exactly today's row, since tomorrow's close is unknown. That's
    correct for building a training/test set (you can't have a training
    example whose label doesn't exist), but it means the LAST row of
    prepare_linear_regression_data()'s X_test is actually YESTERDAY (or
    earlier), not today -- calling predict_next(X_test[-1]) therefore
    predicts a day that's already in `df` and already known (it equals
    df["Close"].iloc[-1], i.e. current_price), while being completely blind
    to the actual most recent day's OHLCV. (Confirmed directly: shocking the
    final row's Close by +500 changed X_test[-1]'s prediction not at all.)
    This function builds the row that was missing -- the one for today,
    which was dropped by dropna() precisely because it has no Target yet.
    """
    if len(df) < 4:
        raise InsufficientDataError(
            "Need at least 4 rows of history to build lag/rolling features for the latest day."
        )
    features = df.copy()
    features["Return_1d"] = features["Close"].pct_change(1)
    features["MA_5"] = features["Close"].rolling(5).mean()
    features["MA_10"] = features["Close"].rolling(10).mean()
    features["Volatility_5"] = features["Close"].rolling(5).std()
    features["Lag_1"] = features["Close"].shift(1)
    features["Lag_2"] = features["Close"].shift(2)
    features["Lag_3"] = features["Close"].shift(3)
    last_row = features[LR_FEATURE_COLUMNS].iloc[-1]
    if last_row.isna().any():
        raise InsufficientDataError(
            "Not enough history to compute all rolling/lag features for the latest day "
            "(need at least 10 rows for a 10-day moving average)."
        )
    return last_row.to_numpy(dtype=float)


def build_latest_lstm_window(df: pd.DataFrame, scaler, lookback: int, target_mode: str):
    """
    Builds the lookback window ending at the MOST RECENT day in `df` (today),
    for a genuine next-day (tomorrow) LSTM forecast -- the LSTM counterpart
    to build_latest_feature_row() above, and fixing the exact same class of
    bug: the LSTM's test sequences (from prepare_lstm_data /
    prepare_lstm_return_data) have their last sequence's TARGET equal to
    today's already-known close, so using that last sequence's window for
    "predict_next" is blind to today's actual value in the same way.

    Uses the SAME already-fitted (train-only) scaler to transform -- it is
    never re-fit here, only applied, so this stays leakage-safe: nothing
    about today's price influences the scaler's own parameters.

    Returns (window_scaled, base_price):
      - window_scaled: shape (lookback,), ready for LSTMPredictor.predict_next()
      - base_price: today's actual close, needed to reconstruct a price
        prediction from a predicted RETURN. None when target_mode == "price"
        (not needed there -- predict_next returns a price directly).
    """
    close = df["Close"].values
    target_mode = (target_mode or "return").strip().lower()

    if target_mode == "return":
        if len(close) < lookback + 1:
            raise InsufficientDataError(
                f"Need at least {lookback + 1} rows of Close prices to build a "
                f"{lookback}-day return window for the latest day."
            )
        returns = (close[1:] - close[:-1]) / close[:-1]
        window_raw = returns[-lookback:].reshape(-1, 1)
        window_scaled = scaler.transform(window_raw).flatten()
        base_price = float(close[-1])
        return window_scaled, base_price

    if len(close) < lookback:
        raise InsufficientDataError(
            f"Need at least {lookback} rows of Close prices to build the latest window."
        )
    window_raw = close[-lookback:].reshape(-1, 1)
    window_scaled = scaler.transform(window_raw).flatten()
    return window_scaled, None


def window_scaled_series(train_scaled: np.ndarray, test_scaled: np.ndarray, lookback: int):
    """
    Pure windowing step (no scaler fit/transform here -- the caller has
    already scaled both arrays). Shared by build_lstm_sequences_leakage_safe
    (which fits a fresh scaler on train_raw first) and by pipeline.py's
    checkpoint-loading path (which reuses an already-fitted scaler loaded
    from disk, and must never re-fit it -- a loaded checkpoint's scaler
    parameters are part of what makes its predictions meaningful).

    Returns: X_train, y_train, X_test, y_test (all as plain windowed arrays;
    reshape to (n, lookback, 1) is left to the caller).
    """
    X_train, y_train = [], []
    for i in range(lookback, len(train_scaled)):
        X_train.append(train_scaled[i - lookback:i, 0])
        y_train.append(train_scaled[i, 0])

    context_tail = train_scaled[-lookback:]
    combined_test = np.concatenate([context_tail, test_scaled], axis=0)
    X_test, y_test = [], []
    for i in range(lookback, len(combined_test)):
        X_test.append(combined_test[i - lookback:i, 0])
        y_test.append(combined_test[i, 0])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def build_lstm_sequences_leakage_safe(train_raw: np.ndarray, test_raw: np.ndarray, lookback: int):
    """
    Shared leakage-safe scaling + windowing core used by both
    prepare_lstm_data() (single train/test split) and validation.py's
    walk-forward validation (many expanding-window folds), so both paths
    follow the exact same "fit scaler on train only" rule instead of two
    subtly different implementations drifting apart over time.

    train_raw / test_raw: 2D arrays of shape (n, 1) of raw (unscaled) closing
    prices, already split chronologically by the caller.

    Returns: X_train, y_train, X_test, y_test, scaler
    """
    if len(train_raw) <= lookback:
        raise InsufficientDataError(
            f"Training portion has only {len(train_raw)} rows, but LOOKBACK_WINDOW="
            f"{lookback} requires more than that just to build a single training "
            f"sequence."
        )
    if len(test_raw) < 1:
        raise InsufficientDataError("Test portion is empty -- nothing to validate against.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)

    train_scaled = scaler.transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train, X_test, y_test = window_scaled_series(train_scaled, test_scaled, lookback)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaler


def prepare_lstm_data(df: pd.DataFrame, lookback: int = None, split_ratio: float = None):
    """
    Builds sliding-window sequences of scaled closing prices for the LSTM.
    Returns X_train, X_test, y_train, y_test, scaler, dates_test

    LEAKAGE-SAFE PIPELINE (important):
    Earlier versions of this function fit the MinMaxScaler on the FULL
    price series (train + test) before splitting -- that lets the scaler's
    min/max be influenced by future (test-period) prices, which then leak
    into every training sequence via the scaled values. That is a real
    data-leakage bug for time series.

    Correct order, followed here:
      1. Split the RAW closing prices chronologically first (no shuffling).
      2. Fit MinMaxScaler ONLY on the training portion.
      3. Transform the training portion with that scaler.
      4. Transform the test portion with the SAME (already-fitted) scaler
         -- the scaler's parameters never see test data.
      5. Build sliding-window sequences from the scaled train/test arrays.

    One subtlety: to produce a prediction for the *first* test-set day, the
    LSTM needs `lookback` prior days of context. Those prior days are the
    tail end of the training period -- real historical data that legitimately
    precedes every test point, so using them is not leakage (nothing from
    the test period is used to build them, and they were scaled using the
    train-only scaler). This mirrors how the model would actually be used
    in production: to predict tomorrow, you always look at some number of
    real past days, some of which will always be "older than your last
    retrain".
    """
    lookback = lookback or config.LOOKBACK_WINDOW
    split_ratio = split_ratio or config.TRAIN_TEST_SPLIT

    close_prices = df["Close"].values.reshape(-1, 1)
    dates = df.index

    split_idx = int(len(close_prices) * split_ratio)
    train_raw = close_prices[:split_idx]
    test_raw = close_prices[split_idx:]
    test_dates = dates[split_idx:]

    X_train, y_train, X_test, y_test, scaler = build_lstm_sequences_leakage_safe(
        train_raw, test_raw, lookback
    )
    seq_dates_test = list(test_dates)  # one date per test sequence (see helper docstring)

    return X_train, X_test, y_train, y_test, scaler, seq_dates_test


def prepare_lstm_return_data(df: pd.DataFrame, lookback: int = None, split_ratio: float = None):
    """
    Alternative to prepare_lstm_data(): instead of mapping a window of past
    RAW PRICES to the next raw price, this maps a window of past daily
    RETURNS (% change) to the next day's return.

    Motivation: raw closing prices are non-stationary -- their level drifts
    over time, so whatever numeric range the MinMaxScaler saw during
    training (necessarily train-only, to stay leakage-safe) can be a poor
    match for the test period's price range whenever the stock has trended
    up or down since training. The LSTM's internal gates (sigmoid/tanh) are
    saturating, so asking the network to extrapolate to scaled values well
    outside [0, 1] (which legitimately happens after a leakage-safe,
    train-only scaler fit -- see prepare_lstm_data's docstring) can produce
    materially worse predictions than a model that never has to extrapolate
    that way in the first place. Daily returns don't drift the same way: a
    stock's typical day-to-day % move tends to stay in a broadly similar
    range whether the stock is at $50 or $500, so a return-based target is
    more stationary and less prone to this specific failure mode.

    Same leakage-safe methodology as prepare_lstm_data: returns are computed
    from the raw close series first, then split chronologically, then the
    scaler is fit ONLY on the training portion's returns.

    Returns: X_train, X_test, y_train, y_test, scaler, dates_test, base_prices_test
    `base_prices_test[i]` is the actual close price the day BEFORE
    dates_test[i] -- i.e. the price a predicted return should be applied to
    (predicted_price = base_price * (1 + predicted_return)) to reconstruct
    a price prediction comparable to prepare_lstm_data's output.
    """
    lookback = lookback or config.LOOKBACK_WINDOW
    split_ratio = split_ratio or config.TRAIN_TEST_SPLIT

    close = df["Close"].values
    if len(close) < 2:
        raise InsufficientDataError("Need at least 2 rows of Close prices to compute returns.")
    returns = (close[1:] - close[:-1]) / close[:-1]
    returns = returns.reshape(-1, 1)
    # returns[i] is the % move FROM day i TO day i+1, i.e. it "belongs" to
    # date[i+1] (the day it was realized on) -- align dates/base-prices
    # accordingly so nothing here accidentally looks one day into the future.
    return_dates = df.index[1:]
    base_prices = close[:-1]  # base_prices[i] is the price the return at return_dates[i] was computed FROM

    split_idx = int(len(returns) * split_ratio)
    train_raw = returns[:split_idx]
    test_raw = returns[split_idx:]
    test_dates = return_dates[split_idx:]
    test_base_prices = base_prices[split_idx:]

    X_train, y_train, X_test, y_test, scaler = build_lstm_sequences_leakage_safe(
        train_raw, test_raw, lookback
    )
    seq_dates_test = list(test_dates)
    seq_base_prices_test = list(test_base_prices)

    return X_train, X_test, y_train, y_test, scaler, seq_dates_test, seq_base_prices_test



def perf_decorated(name):
    """Record the wall time of the decorated function under `name` (see perf.py)."""
    import functools
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*a, **k):
            with perf.timed(name):
                return fn(*a, **k)
        return inner
    return wrap


@perf_decorated("pipeline.feature_engineering_lstm")
def prepare_lstm_training_data(df: pd.DataFrame, target_mode: str = None,
                                lookback: int = None, split_ratio: float = None):
    """
    Unified entry point dispatching to prepare_lstm_data ("price" mode) or
    prepare_lstm_return_data ("return" mode), always returning the same
    7-tuple shape so callers (pipeline.py, validation.py) don't need mode-
    specific branches: X_train, X_test, y_train, y_test, scaler, dates_test,
    base_prices_test (the last element is None in "price" mode, since no
    price reconstruction step is needed there).

    Defaults to config.LSTM_PREDICTION_TARGET -- see that setting's
    docstring in config.py for the empirical comparison behind defaulting
    to "return".
    """
    target_mode = (target_mode or config.LSTM_PREDICTION_TARGET).strip().lower()
    if target_mode == "return":
        return prepare_lstm_return_data(df, lookback=lookback, split_ratio=split_ratio)
    if target_mode == "price":
        X_train, X_test, y_train, y_test, scaler, dates_test = prepare_lstm_data(
            df, lookback=lookback, split_ratio=split_ratio
        )
        return X_train, X_test, y_train, y_test, scaler, dates_test, None
    raise ValueError(f"Unknown LSTM target_mode '{target_mode}' -- expected 'return' or 'price'.")


def recent_trend_summary(df: pd.DataFrame, ticker: str = None, days: int = 15) -> str:
    """
    Produces a compact plain-text summary of recent price action, used as the
    prompt context sent to each LLM (they don't get raw numerical arrays --
    just a readable digest, since that's what a language model reasons over
    best).

    `ticker` should always be the actual runtime-selected ticker (e.g. what
    was passed to --ticker / the webapp form), NOT read implicitly from
    config.TICKER. This used to default silently to config.TICKER, which
    meant `python main.py --ticker TSLA` could send the LLM panel a prompt
    saying "Ticker: AAPL" whenever config.TICKER (the .env default) still
    said AAPL -- the model never actually saw which stock it was analyzing.
    The `ticker` parameter is only optional here for the rare case of
    calling this function directly/interactively without a pipeline run;
    every real call site in this project (pipeline.py) always passes it
    explicitly.
    """
    ticker_label = ticker if ticker else config.TICKER
    recent = df.tail(days)
    start_price = float(recent["Close"].iloc[0])
    end_price = float(recent["Close"].iloc[-1])
    pct_change = (end_price - start_price) / start_price * 100
    high = float(recent["High"].max())
    low = float(recent["Low"].min())
    avg_vol = float(recent["Volume"].mean())

    summary = (
        f"Ticker: {ticker_label}\n"
        f"Last {days} trading days: from ${start_price:.2f} to ${end_price:.2f} "
        f"({pct_change:+.2f}%)\n"
        f"Period high: ${high:.2f}, period low: ${low:.2f}\n"
        f"Average daily volume: {avg_vol:,.0f}\n"
        f"Most recent 5 closes: {[round(x, 2) for x in recent['Close'].tail(5).tolist()]}"
    )
    return summary


if __name__ == "__main__":
    data = fetch_stock_data()
    print(data.tail())
    print("\n--- Trend summary for LLM prompts ---")
    print(recent_trend_summary(data))
