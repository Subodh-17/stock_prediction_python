"""
Integration test: pipeline.run_pipeline() (mocked data fetch, synthetic
prices) feeding directly into trading.trade_executor.execute_trade_decision()
in paper mode -- validates the two halves of the project (prediction and
trading) actually wire together correctly end to end, the way trade.py uses
them for real.
"""
import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

import config

TMP_DIR = tempfile.mkdtemp()
config.TRADE_LOG_PATH = os.path.join(TMP_DIR, "trade_log.csv")
config.PAPER_PORTFOLIO_PATH = os.path.join(TMP_DIR, "paper_portfolio.json")
config.PAPER_STARTING_CASH = 100000

import pipeline
from trading.trade_executor import execute_trade_decision

np.random.seed(7)
n = 400
dates = pd.bdate_range("2024-01-01", periods=n)
returns = np.random.normal(loc=0.0008, scale=0.012, size=n)  # slight upward drift
close = 200 * np.exp(np.cumsum(returns))
high = close * (1 + np.abs(np.random.normal(0, 0.004, n)))
low = close * (1 - np.abs(np.random.normal(0, 0.004, n)))
open_ = close * (1 + np.random.normal(0, 0.002, n))
volume = np.random.randint(1_000_000, 5_000_000, n)
synthetic_df = pd.DataFrame(
    {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates
)

with patch("data_loader.fetch_stock_data", return_value=synthetic_df):
    result = pipeline.run_pipeline(
        ticker="SYNTH", period="2y", epochs=3, source="yfinance", skip_llm=True, verbose=False,
    )

assert "final_ensemble" in result
assert result["final_ensemble"]["current_price"] > 0
print("Test 1 (pipeline.run_pipeline end-to-end with mocked data fetch) passed:",
      result["final_ensemble"])

outcome = execute_trade_decision(
    ticker="SYNTH", final_ensemble=result["final_ensemble"], mode="paper", verbose=False,
)
action = outcome["signal"]["action"]
assert action in ("BUY", "SELL", "HOLD")
print("Test 2 (trade decision generated from real pipeline output) passed:", action)

if action == "BUY":
    # A fresh paper account has starting cash, so a BUY signal should execute.
    assert outcome["executed"] is True, outcome
    assert outcome["quantity"] > 0
    print("Test 3 (BUY signal executes against paper broker with valid quantity) passed:",
          outcome["quantity"])
elif action == "SELL":
    # A fresh paper account holds 0 shares, so a SELL signal should be
    # correctly BLOCKED (there's nothing to sell) -- not silently executed
    # as a phantom short-sell. This is the fixed behavior, not a bug.
    assert outcome["executed"] is False, outcome
    assert "no position" in outcome["reason"].lower() or "held" in outcome["reason"].lower()
    print("Test 3 (SELL signal on empty holdings correctly BLOCKED, not phantom-executed) passed:",
          outcome["reason"])
else:
    print("Test 3 skipped (signal was HOLD for this random seed -- not an error)")

# confirm it was actually logged, and that the daily trade count only
# reflects genuinely EXECUTED trades (not a blocked SELL attempt)
from trading import trade_log
trades_today = trade_log.count_trades_today()
expected_count = 1 if outcome["executed"] else 0
assert trades_today == expected_count, (trades_today, expected_count, outcome)
print("Test 4 (trade decision correctly reflected in the audit log) passed")

import shutil
shutil.rmtree(TMP_DIR, ignore_errors=True)
print("\nALL INTEGRATION TESTS PASSED")
