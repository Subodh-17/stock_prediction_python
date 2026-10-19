"""
Test harness using synthetic OHLCV data (since this sandbox has no internet
access to Yahoo Finance). Validates that data_loader, both models, the
ensemble, and visualize all work correctly end-to-end before shipping.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

import data_loader
from models.linear_regression_model import LinearRegressionPredictor
from models.lstm_model import LSTMPredictor
from ai_integrations.panel import query_llm_panel
import ensemble
import visualize
import config

np.random.seed(42)

# ---- build a realistic-looking synthetic price series (random walk + drift) ----
n = 500
dates = pd.bdate_range("2023-01-01", periods=n)
returns = np.random.normal(loc=0.0005, scale=0.015, size=n)
close = 150 * np.exp(np.cumsum(returns))
high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
open_ = close * (1 + np.random.normal(0, 0.003, n))
volume = np.random.randint(1_000_000, 5_000_000, n)

df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
config.TICKER = "SYNTH"
current_price = float(df["Close"].iloc[-1])
print(f"Synthetic data built: {df.shape}, current price ${current_price:.2f}")

# ---- Linear Regression ----
X_train, X_test, y_train, y_test, feature_names, lr_dates_test = data_loader.prepare_linear_regression_data(df)
lr_model = LinearRegressionPredictor().train(X_train, y_train, feature_names)
lr_eval = lr_model.evaluate(X_test, y_test)
print(f"LR  -> RMSE {lr_eval['rmse']:.3f}  R2 {lr_eval['r2']:.3f}  MAPE {lr_eval['mape']:.2f}%")
lr_next_price = lr_model.predict_next(X_test[-1])
print(f"LR next-day prediction: ${lr_next_price:.2f}")

# ---- LSTM (small epoch count just for a smoke test) ----
lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test, scaler, lstm_dates_test = data_loader.prepare_lstm_data(df)
lstm_model = LSTMPredictor()
lstm_model.train(lstm_X_train, lstm_y_train, epochs=3, verbose=0)
lstm_eval = lstm_model.evaluate(lstm_X_test, lstm_y_test, scaler=scaler)
print(f"LSTM -> RMSE {lstm_eval['rmse']:.3f}  R2 {lstm_eval['r2']:.3f}  MAPE {lstm_eval['mape']:.2f}%")
last_window = lstm_X_test[-1].flatten()
lstm_next_price = lstm_model.predict_next(last_window, scaler)
print(f"LSTM next-day prediction: ${lstm_next_price:.2f}")

# ---- LLM panel (no keys set -> should gracefully skip) ----
trend_summary = data_loader.recent_trend_summary(df)
print("\n--- trend summary sent to LLMs ---")
print(trend_summary)
llm_results = query_llm_panel(trend_summary)
llm_panel_summary = ensemble.blend_llm_panel(llm_results, current_price)
print("LLM panel summary (should be None with no keys):", llm_panel_summary)

# ---- Ensemble ----
final = ensemble.compute_ensemble_prediction(lr_next_price, lstm_next_price, llm_panel_summary, current_price)
print("\nFinal ensemble:", final)

# ---- Visualization ----
chart1 = visualize.plot_model_comparison(lr_dates_test, y_test, lr_eval["predictions"], lstm_dates_test, lstm_eval["predictions"])
chart3 = visualize.plot_final_ensemble(final)
print(f"\nCharts saved: {chart1}, {chart3}")

# ---- Insufficient data guard (regression test for a real crash found during
# QA: a short --period combined with the default LOOKBACK_WINDOW used to
# crash deep inside numpy with a confusing "IndexError: tuple index out of
# range" instead of a clear, actionable message) ----
short_dates = pd.bdate_range("2026-06-01", periods=20)
short_df = pd.DataFrame(
    {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000},
    index=short_dates,
)
try:
    data_loader.validate_sufficient_data(short_df)
    raise AssertionError("Should have raised InsufficientDataError for 20 rows with lookback=30")
except data_loader.InsufficientDataError as e:
    assert "20 rows" in str(e)
    print("Insufficient-data guard passed: raised a clear error instead of crashing downstream:", e)

print("\nALL TESTS PASSED")
