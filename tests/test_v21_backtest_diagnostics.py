"""Regression tests for Quant Workstation V21: Backtest Signal Diagnostics and Position Sizing Normalization."""
import pytest
import numpy as np
import backtest as bt

DATES = [f"2026-05-{i:02d}" for i in range(1, 11)]
CLOSE = [100.0, 100.5, 101.0, 100.8, 101.2, 101.5, 101.3, 101.8, 102.0, 102.2]


def test_zero_trade_generates_truthful_signal_diagnostics():
    """When a model's predicted returns are too small to clear the BUY threshold,
    signal_diagnostics must explain the exact min/max predicted returns and why 0 trades occurred."""
    # Forecasts only +0.1% to +0.2% every day -> strictly below default +0.50% buy threshold
    pred = [100.0] + [c * 1.002 for c in CLOSE[:-1]]
    
    r = bt.run_backtest(DATES, CLOSE, pred, buy_threshold_pct=0.5, sell_threshold_pct=-0.5)
    assert r.num_trades == 0
    assert r.total_return_pct == 0.0
    assert r.exposure_pct == 0.0
    
    diag = r.signal_diagnostics
    assert diag is not None
    assert diag["buy_signals"] == 0
    assert diag["sell_signals"] == 0
    assert diag["hold_signals"] == len(DATES) - 1
    assert diag["max_predicted_pct_change"] < 0.50
    assert "never reaching the Buy threshold" in diag["zero_trade_reason"]
    assert "Zero-trade explanation:" in r.summary_text()


def test_position_size_pct_normalization():
    """If a caller passes position_size_pct=100 (intending 100%), it must be normalized
    to 1.0 rather than trading on 100x margin."""
    # Forecasts +1.0% on day 1 (triggering BUY) and -1.0% on day 3 (triggering SELL)
    pred = [100.0, 101.0, 101.0, 99.0] + [100.0] * 6
    
    r = bt.run_backtest(DATES, CLOSE, pred, initial_capital=10000, buy_threshold_pct=0.5,
                        sell_threshold_pct=-0.5, position_size_pct=100.0)
    assert r.position_size_pct == 1.0
    # First buy trade notional should be ~10,000, not 1,000,000
    assert len(r.trades) >= 1
    buy_trade = r.trades[0]
    assert buy_trade.shares * buy_trade.price <= 10001.0
