"""
metrics_utils.py
------------------
Shared metric helpers used by both models (and the webapp/CLI reporting
layers), kept in one place so "Directional Accuracy (%)" means exactly the
same thing everywhere it's shown.

NAMING: this module measures DIRECTIONAL correctness -- whether the
predicted move was up or down -- not classification accuracy and not
"how accurate the model is" in any general sense. Every user-facing label
for it reads "Directional Accuracy (%)" for that reason. The dict key
stays `accuracy_pct` for backward compatibility with existing report.json
files and API consumers; API responses additionally expose it under the
unambiguous name `directional_accuracy_pct`.
"""

import numpy as np


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    "Directional Accuracy (%)" -- the percentage of test-set days where the
    model correctly predicted the DIRECTION of the next move (up or down),
    not just how close the price number was.

    This is NOT classification accuracy and must never be labeled as plain
    "accuracy" in any user-facing surface: it says nothing about whether the
    predicted PRICE was close (that is RMSE/MAE/MAPE), and a model can score
    well here while forecasting prices poorly, or vice versa.

    For test-set index i (i >= 1): the model's predicted direction is
    sign(y_pred[i] - y_true[i-1]) -- did it predict a move up or down from
    the last known actual price -- compared against the real outcome,
    sign(y_true[i] - y_true[i-1]). This is exactly the same "predicted
    change relative to the last known price" logic used everywhere else in
    this project (signal_generator.py, the backtest simulator), so this
    number is consistent with what actually drives BUY/SELL/HOLD decisions,
    not a separately-invented definition of directional correctness.

    A flat (zero) real-world move counts as a "down" for comparison
    purposes (np.sign(0) == 0, which only matches a predicted 0 exactly) --
    an edge case rare enough in real price data not to matter in practice.

    Returns a percentage in [0, 100]. Needs at least 2 points; returns NaN
    for shorter series rather than dividing by zero.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)
    if n < 2 or len(y_pred) != n:
        return float("nan")

    actual_direction = np.sign(y_true[1:] - y_true[:-1])
    predicted_direction = np.sign(y_pred[1:] - y_true[:-1])
    matches = np.sum(actual_direction == predicted_direction)
    return float(matches) / (n - 1) * 100


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error, excluding observations whose ACTUAL value
    is zero.

    Plain MAPE divides by the actual value, so a single zero price makes the
    whole metric +inf -- verified directly. That inf then propagates into
    report.json, the CSV/PDF exports and the API response as though it were a
    measurement. A zero close is not normal market data (check_data_quality
    flags it as an ISSUE), but the pipeline deliberately proceeds on flagged
    data rather than blocking, so the metric has to cope.

    Returns NaN when NO observation has a usable (non-zero) actual value --
    honestly "not measurable" rather than a number computed from nothing.
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    usable = y_true != 0
    if not np.any(usable):
        return float("nan")
    return float(np.mean(np.abs((y_true[usable] - y_pred[usable]) / y_true[usable])) * 100)
