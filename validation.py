"""
validation.py
---------------
Walk-forward (expanding-window) validation for the LR and LSTM models, on
top of the existing single train/test split evaluation (which is NOT
removed or replaced -- this is an additional, more robust evaluation
method, as requested).

Why this exists:
A single chronological train/test split (what data_loader.prepare_*_data
already does) tells you how the model performs on ONE held-out period. It
can be lucky or unlucky depending on what that period happened to look
like. Walk-forward validation instead re-evaluates the model across several
sequential, expanding training windows -- never shuffled, never using
future data to predict the past -- and reports the distribution of
performance across folds, which is a much more honest picture of how the
model would have performed if you'd actually been retraining and trading
with it over time.

Fold layout (expanding window, N folds):

    Fold 1: train [0 .......... a)      validate [a ..... b)
    Fold 2: train [0 ................. b)  validate [b ..... c)
    Fold 3: train [0 ...................... c) validate [c ..... d)
    ...

Each fold's validation window is completely in the future relative to
that fold's training window, and training data only ever grows forward in
time -- it is never shuffled and never includes rows from a fold's own
validation window or any later window.

This module deliberately keeps LSTM walk-forward opt-in / cheaper than the
main LSTM (fewer default epochs, no early-stopping patience blow-up) since
retraining a neural net several times can be slow; that tradeoff is
documented on the function itself.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import config
import data_loader
from metrics_utils import directional_accuracy
from models.linear_regression_model import LinearRegressionPredictor
from models.lstm_model import LSTMPredictor


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    train_end_date: str
    test_start_date: str
    test_end_date: str
    mae: float
    rmse: float
    r2: Optional[float]
    directional_accuracy_pct: float


@dataclass
class WalkForwardResult:
    model_name: str
    n_folds_requested: int
    folds: List[FoldResult] = field(default_factory=list)
    mean_mae: float = float("nan")
    mean_rmse: float = float("nan")
    mean_r2: Optional[float] = None
    mean_directional_accuracy_pct: float = float("nan")
    std_rmse: float = float("nan")

    @property
    def n_folds(self) -> int:
        """
        The number of folds that actually produced a usable result -- NOT
        the number requested/attempted. A short series can cause some
        candidate folds to be skipped (too few test rows, or too little
        training data for the lookback window); those never make it into
        `folds`, so n_folds can legitimately be lower than n_folds_requested.
        Reporting the requested count here instead of the successful count
        was a real bug: it made a walk-forward run with, say, 0 successful
        folds out of 2 requested print "over 2 folds" alongside a silently
        NaN mean RMSE, which reads as if 2 folds were evaluated and simply
        produced a bad number, rather than "nothing could actually be
        evaluated." (compute_dynamic_weights' own safety check already
        correctly used len(folds) rather than this field, so this was a
        reporting/transparency bug, not a bug that let bad data through to
        the ensemble.)
        """
        return len(self.folds)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_folds_requested": self.n_folds_requested,
            "n_folds": self.n_folds,
            "folds": [vars(f) for f in self.folds],
            "mean_mae": self.mean_mae,
            "mean_rmse": self.mean_rmse,
            "std_rmse": self.std_rmse,
            "mean_r2": self.mean_r2,
            "mean_directional_accuracy_pct": self.mean_directional_accuracy_pct,
        }


def _make_fold_boundaries(n_rows: int, n_splits: int, min_train_size: int):
    """
    Returns a list of (train_end, test_end) index pairs for an expanding
    window split. train_end grows each fold; the validation slice for fold k
    is rows [train_end_k : test_end_k).
    """
    remaining = n_rows - min_train_size
    if remaining < n_splits:
        n_splits = max(1, remaining)
    fold_size = remaining // n_splits
    if fold_size < 1:
        raise data_loader.InsufficientDataError(
            f"Not enough rows ({n_rows}) to build {n_splits} walk-forward folds "
            f"with a minimum training size of {min_train_size}."
        )
    boundaries = []
    train_end = min_train_size
    for k in range(n_splits):
        test_end = train_end + fold_size if k < n_splits - 1 else n_rows
        boundaries.append((train_end, test_end))
        train_end = test_end
    return boundaries


def walk_forward_validate_lr(df, n_splits: int = 5, min_train_size: int = None) -> WalkForwardResult:
    """Expanding-window walk-forward validation for the Linear Regression model."""
    featured = data_loader.add_technical_features(df)
    feature_cols = data_loader.LR_FEATURE_COLUMNS
    X = featured[feature_cols].values
    y = featured["Target"].values
    # Fold boundaries are labeled by TARGET date (the day each y value was
    # realized), not by the day its features were observed -- see
    # data_loader.lr_target_dates(). This matters beyond cosmetics:
    # backtest._compute_historical_dynamic_weights() only admits folds whose
    # test_end_date falls strictly BEFORE the backtest day being weighted.
    # With feature dates, a fold that had in fact already observed day d's
    # close reported test_end_date = d-1 and was therefore admitted when
    # weighting day d -- a one-day look-ahead. Target dates close that.
    dates = data_loader.lr_target_dates(df, featured.index)

    n = len(X)
    min_train_size = min_train_size or max(30, n // (n_splits + 1))
    boundaries = _make_fold_boundaries(n, n_splits, min_train_size)

    result = WalkForwardResult(model_name="LinearRegression", n_folds_requested=len(boundaries))
    for k, (train_end, test_end) in enumerate(boundaries, start=1):
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        if len(X_test) < 2:
            continue  # need >=2 points for directional accuracy

        model = LinearRegressionPredictor()
        model.train(X_train, y_train, feature_names=feature_cols)
        ev = model.evaluate(X_test, y_test)

        result.folds.append(FoldResult(
            fold=k,
            train_size=len(X_train),
            test_size=len(X_test),
            train_end_date=str(dates[train_end - 1].date()),
            test_start_date=str(dates[train_end].date()),
            test_end_date=str(dates[test_end - 1].date()),
            mae=ev["mae"],
            rmse=ev["rmse"],
            r2=ev["r2"],
            directional_accuracy_pct=ev["accuracy_pct"],
        ))

    _aggregate(result)
    return result


def walk_forward_validate_lstm(df, n_splits: int = 3, min_train_size: int = None,
                                lookback: int = None, epochs: int = 8,
                                target_mode: str = None) -> WalkForwardResult:
    """
    Expanding-window walk-forward validation for the LSTM.

    `target_mode`: "return" (default, from config.LSTM_PREDICTION_TARGET) or
    "price" -- IMPORTANT that this matches whatever the main (single-split)
    LSTM actually predicts. If it didn't, the walk-forward RMSE used for
    dynamic ensemble weighting (see ensemble.compute_dynamic_weights) would
    be measuring a different model than the one actually being deployed,
    which would make the resulting weights misleading rather than a fair
    reflection of the real model's out-of-sample performance.

    NOTE on cost: this retrains a full LSTM once per fold. To keep total
    runtime reasonable, this defaults to fewer folds (3) and fewer epochs
    (8, vs. config.LSTM_EPOCHS which defaults to 25) than the main
    single-split model. This is a real, documented tradeoff between
    validation thoroughness and runtime -- callers who want a closer match
    to production training can pass a larger `epochs`.

    Each fold still follows the leakage-safe rule: the MinMaxScaler for
    that fold is fit ONLY on that fold's training rows (see
    data_loader.build_lstm_sequences_leakage_safe), never on its own
    validation rows or any future data. In "return" mode, returns
    themselves are computed from the full close series first (a fixed,
    parameter-free transform -- there is nothing to "fit" or leak in a
    percent-change calculation), then split into folds exactly like raw
    prices are in "price" mode; only the scaler fit is restricted to each
    fold's training rows.
    """
    lookback = lookback or config.LOOKBACK_WINDOW
    target_mode = (target_mode or config.LSTM_PREDICTION_TARGET).strip().lower()
    dates = df.index

    if target_mode == "return":
        close = df["Close"].values
        raw_series = ((close[1:] - close[:-1]) / close[:-1]).reshape(-1, 1)
        base_prices_full = close[:-1]
        fold_dates = dates[1:]  # raw_series[i] "belongs to" fold_dates[i]
    else:
        raw_series = df["Close"].values.reshape(-1, 1)
        base_prices_full = None
        fold_dates = dates

    n = len(raw_series)
    min_train_size = min_train_size or max(lookback + 20, n // (n_splits + 2))
    boundaries = _make_fold_boundaries(n, n_splits, min_train_size)

    result = WalkForwardResult(model_name="LSTM", n_folds_requested=len(boundaries))
    for k, (train_end, test_end) in enumerate(boundaries, start=1):
        train_raw = raw_series[:train_end]
        test_raw = raw_series[train_end:test_end]
        if len(test_raw) < 2 or len(train_raw) <= lookback:
            continue

        try:
            X_train, y_train, X_test, y_test, scaler = data_loader.build_lstm_sequences_leakage_safe(
                train_raw, test_raw, lookback
            )
        except data_loader.InsufficientDataError:
            continue

        fold_base_prices = base_prices_full[train_end:test_end] if base_prices_full is not None else None

        model = LSTMPredictor(lookback=lookback)
        model.train(X_train, y_train, epochs=epochs, verbose=0)
        ev = model.evaluate(X_test, y_test, scaler=scaler, base_prices=fold_base_prices)

        result.folds.append(FoldResult(
            fold=k,
            train_size=len(X_train),
            test_size=len(X_test),
            train_end_date=str(fold_dates[train_end - 1].date()),
            test_start_date=str(fold_dates[train_end].date()),
            test_end_date=str(fold_dates[test_end - 1].date()),
            mae=ev["mae"],
            rmse=ev["rmse"],
            r2=ev["r2"],
            directional_accuracy_pct=ev["accuracy_pct"],
        ))

    _aggregate(result)
    return result


def _aggregate(result: WalkForwardResult) -> None:
    if not result.folds:
        return
    # Defensively filter non-finite values out of every metric before
    # averaging (matching how r2/accuracy were already filtered) -- a single
    # degenerate fold (e.g. a numerically unstable fit on very little data)
    # should never silently poison the mean for every other fold that
    # produced a sane result.
    rmses = [f.rmse for f in result.folds if np.isfinite(f.rmse)]
    maes = [f.mae for f in result.folds if np.isfinite(f.mae)]
    r2s = [f.r2 for f in result.folds if f.r2 is not None and np.isfinite(f.r2)]
    accs = [f.directional_accuracy_pct for f in result.folds if np.isfinite(f.directional_accuracy_pct)]

    result.mean_mae = float(np.mean(maes)) if maes else float("nan")
    result.mean_rmse = float(np.mean(rmses)) if rmses else float("nan")
    result.std_rmse = float(np.std(rmses)) if rmses else float("nan")
    result.mean_r2 = float(np.mean(r2s)) if r2s else None
    result.mean_directional_accuracy_pct = float(np.mean(accs)) if accs else float("nan")
