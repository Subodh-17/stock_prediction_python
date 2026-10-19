"""
Comprehensive tests for trading/ -- signal generation, risk limits, the
paper broker (fully tested for real, since it's local/no-network), the
trade log, and the safety-gate logic in trade_executor (mocked where a real
broker/network call would otherwise be needed).
"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from trading import signal_generator, risk_manager, trade_log
from trading.brokers.paper_broker import PaperBroker

# Use isolated temp files for every test so this test never touches real
# project state (outputs/trade_log.csv, .paper_portfolio.json) or leaves
# artifacts behind.
TMP_DIR = tempfile.mkdtemp()
config.TRADE_LOG_PATH = os.path.join(TMP_DIR, "trade_log.csv")
config.PAPER_PORTFOLIO_PATH = os.path.join(TMP_DIR, "paper_portfolio.json")

# ---------------------------------------------------------------------------
# signal_generator
# ---------------------------------------------------------------------------
buy_case = {"predicted_pct_change": 1.5, "final_predicted_price": 203.0, "current_price": 200.0}
sig = signal_generator.generate_signal(buy_case)
assert sig["action"] == "BUY", sig
print("Test 1 (signal: clear BUY) passed:", sig["action"])

sell_case = {"predicted_pct_change": -1.5, "final_predicted_price": 197.0, "current_price": 200.0}
sig = signal_generator.generate_signal(sell_case)
assert sig["action"] == "SELL", sig
print("Test 2 (signal: clear SELL) passed:", sig["action"])

hold_case = {"predicted_pct_change": 0.1, "final_predicted_price": 200.2, "current_price": 200.0}
sig = signal_generator.generate_signal(hold_case)
assert sig["action"] == "HOLD", sig
print("Test 3 (signal: inside band -> HOLD) passed:", sig["action"])

# exact threshold boundary -- should count as a signal, not HOLD (>= / <=)
boundary_case = {"predicted_pct_change": config.BUY_THRESHOLD_PCT, "final_predicted_price": 201.0, "current_price": 200.0}
sig = signal_generator.generate_signal(boundary_case)
assert sig["action"] == "BUY", sig
print("Test 4 (signal: exactly at BUY threshold -> BUY) passed")

try:
    signal_generator.generate_signal(buy_case, buy_threshold=-1, sell_threshold=1)
    raise AssertionError("Should have raised ValueError for inverted thresholds")
except ValueError:
    print("Test 5 (signal: inverted thresholds -> ValueError) passed")

# regression test: a NaN prediction must never silently become a HOLD signal
nan_case = {"predicted_pct_change": float("nan"), "final_predicted_price": float("nan"), "current_price": 200.0}
try:
    signal_generator.generate_signal(nan_case)
    raise AssertionError("Should have raised ValueError for a NaN prediction")
except ValueError as e:
    assert "nan" in str(e).lower()
    print("Test 5b (signal: NaN prediction raises clearly instead of silently -> HOLD) passed")

# ---------------------------------------------------------------------------
# risk_manager: position sizing
# ---------------------------------------------------------------------------
qty = risk_manager.compute_order_quantity(available_capital=100000, current_price=200,
                                           max_position_pct=0.10, max_order_value=10000)
# 10% of 100000 = 10000, capped at max_order_value=10000 too -> 10000/200 = 50 shares
assert qty == 50, qty
print("Test 6 (risk_manager: position sizing, both caps agree) passed:", qty)

qty = risk_manager.compute_order_quantity(available_capital=100000, current_price=200,
                                           max_position_pct=0.01, max_order_value=10000)
# 1% of 100000 = 1000 -> 1000/200 = 5 shares (more restrictive than max_order_value)
assert qty == 5, qty
print("Test 7 (risk_manager: max_position_pct is the binding constraint) passed:", qty)

qty = risk_manager.compute_order_quantity(available_capital=0, current_price=200)
assert qty == 0, qty
print("Test 8 (risk_manager: zero capital -> 0 quantity) passed")

qty = risk_manager.compute_order_quantity(available_capital=50, current_price=1000)
assert qty == 0, qty  # can't afford even 1 share
print("Test 9 (risk_manager: price too high for capital -> 0 quantity) passed")

# ---------------------------------------------------------------------------
# risk_manager: SELL sizing is based on HOLDINGS, never on capital
# (regression test for a real bug found during QA: SELL orders were being
# sized the same way as BUY orders, based on available cash, which let a
# SELL signal attempt to sell shares that were never actually held)
# ---------------------------------------------------------------------------
qty = risk_manager.compute_sell_quantity(held_quantity=0, current_price=200)
assert qty == 0, qty
print("Test 9b (risk_manager: SELL with 0 held -> 0 quantity, regardless of capital) passed")

qty = risk_manager.compute_sell_quantity(held_quantity=30, current_price=200)
assert qty == 30, qty  # defaults to selling the FULL position
print("Test 9c (risk_manager: SELL with no requested_quantity sells full position) passed")

qty = risk_manager.compute_sell_quantity(held_quantity=30, current_price=200, requested_quantity=10)
assert qty == 10, qty
print("Test 9d (risk_manager: SELL respects a smaller requested_quantity) passed")

qty = risk_manager.compute_sell_quantity(held_quantity=30, current_price=200, requested_quantity=999)
assert qty == 30, qty  # can never exceed what's actually held
print("Test 9e (risk_manager: SELL requested_quantity capped at held_quantity) passed")

try:
    risk_manager.validate_order(quantity=50, current_price=200, available_capital=0,
                                 action="SELL", held_quantity=10)
    raise AssertionError("Should have raised -- trying to sell more than held")
except risk_manager.RiskLimitError:
    print("Test 9f (risk_manager: validate_order rejects SELL exceeding held_quantity) passed")

# A SELL with 0 available_capital should still validate fine as long as
# held_quantity covers it -- SELL orders don't need capital, they need shares.
risk_manager.validate_order(quantity=10, current_price=200, available_capital=0,
                             action="SELL", held_quantity=10)
print("Test 9g (risk_manager: SELL validates fine with 0 capital, as long as shares are held) passed")

try:
    risk_manager.validate_order(quantity=1000, current_price=200, available_capital=0,
                                 action="SELL", held_quantity=1000)
    raise AssertionError("Should have raised -- order value exceeds MAX_ORDER_VALUE")
except risk_manager.RiskLimitError:
    print("Test 9h (risk_manager: SELL still capped by MAX_ORDER_VALUE) passed")

try:
    risk_manager.validate_order(quantity=1000, current_price=200, available_capital=100000,
                                 max_position_pct=0.10, max_order_value=10000)
    raise AssertionError("Should have raised RiskLimitError -- order exceeds cap")
except risk_manager.RiskLimitError:
    print("Test 10 (risk_manager: validate_order rejects oversized order) passed")

risk_manager.validate_order(quantity=50, current_price=200, available_capital=100000,
                             max_position_pct=0.10, max_order_value=10000)
print("Test 11 (risk_manager: validate_order accepts a compliant order) passed")

# ---------------------------------------------------------------------------
# trade_log + daily trade limit
# ---------------------------------------------------------------------------
assert trade_log.count_trades_today() == 0
trade_log.log_trade(ticker="TEST", action="BUY", mode="paper", broker="paper",
                     predicted_pct_change=1.0, current_price=100, predicted_price=101,
                     quantity=10, order_value=1000, status="FILLED")
assert trade_log.count_trades_today() == 1
trade_log.log_trade(ticker="TEST", action="HOLD", mode="paper", broker="paper",
                     predicted_pct_change=0.1, current_price=100, predicted_price=100.1)
# HOLD shouldn't count toward the daily trade limit
assert trade_log.count_trades_today() == 1
print("Test 12 (trade_log: counts BUY/SELL but not HOLD toward daily limit) passed")

try:
    risk_manager.check_daily_trade_limit(max_trades_per_day=1)
    raise AssertionError("Should have raised RiskLimitError -- limit already hit")
except risk_manager.RiskLimitError:
    print("Test 13 (risk_manager: daily trade limit enforced) passed")

risk_manager.check_daily_trade_limit(max_trades_per_day=5)  # should not raise
print("Test 14 (risk_manager: daily trade limit allows under cap) passed")

# ---------------------------------------------------------------------------
# PaperBroker: full simulated buy/sell lifecycle
# ---------------------------------------------------------------------------
broker = PaperBroker(portfolio_path=os.path.join(TMP_DIR, "portfolio2.json"), starting_cash=10000)
assert broker.get_available_capital() == 10000
assert broker.get_position("AAPL") == 0

result = broker.place_order("AAPL", "BUY", 10, price=100)
assert result["status"] == "FILLED", result
assert broker.get_available_capital() == 9000
assert broker.get_position("AAPL") == 10
print("Test 15 (PaperBroker: BUY reduces cash, increases holdings) passed")

result = broker.place_order("AAPL", "BUY", 1000, price=100)  # way more than remaining cash
assert result["status"] == "REJECTED", result
assert "Insufficient" in result["message"]
print("Test 16 (PaperBroker: BUY beyond available cash -> REJECTED) passed")

result = broker.place_order("AAPL", "SELL", 5, price=110)
assert result["status"] == "FILLED", result
assert broker.get_position("AAPL") == 5
assert broker.get_available_capital() == 9000 + 550
print("Test 17 (PaperBroker: SELL increases cash, decreases holdings) passed")

result = broker.place_order("AAPL", "SELL", 999, price=110)  # more than held
assert result["status"] == "REJECTED", result
assert "Insufficient" in result["message"]
print("Test 18 (PaperBroker: SELL beyond holdings -> REJECTED) passed")

result = broker.place_order("AAPL", "SELL", 5, price=110)  # sell the rest
assert result["status"] == "FILLED", result
assert broker.get_position("AAPL") == 0  # fully closed position removed from holdings
print("Test 19 (PaperBroker: fully closing a position works cleanly) passed")

result = broker.place_order("AAPL", "BUY", 0, price=100)
assert result["status"] == "REJECTED"
print("Test 20 (PaperBroker: zero quantity -> REJECTED) passed")

result = broker.place_order("AAPL", "HOLD", 5, price=100)
assert result["status"] == "REJECTED"
print("Test 21 (PaperBroker: unknown action -> REJECTED) passed")

# ---------------------------------------------------------------------------
# PaperBroker: state persists across instances (reads back from disk)
# ---------------------------------------------------------------------------
broker2 = PaperBroker(portfolio_path=os.path.join(TMP_DIR, "portfolio2.json"))
assert broker2.get_available_capital() == broker.get_available_capital()
print("Test 22 (PaperBroker: state persists to disk across instances) passed")

# ---------------------------------------------------------------------------
# trade_executor: safety gates (mocked broker calls not needed -- paper path
# exercises the real logic; live path is tested via the ENABLE_LIVE_TRADING gate)
# ---------------------------------------------------------------------------
from trading.trade_executor import execute_trade_decision

# HOLD signal -> no broker call at all, no order attempted
hold_ensemble = {"predicted_pct_change": 0.0, "final_predicted_price": 200.0, "current_price": 200.0}
outcome = execute_trade_decision("TESTX", hold_ensemble, mode="paper", verbose=False)
assert outcome["executed"] is False
assert outcome["signal"]["action"] == "HOLD"
print("Test 23 (trade_executor: HOLD signal never reaches a broker) passed")

# regression test: a NaN prediction reaching trade_executor must be blocked
# cleanly with a log entry, not crash with no record of the attempt
nan_ensemble = {"predicted_pct_change": float("nan"), "final_predicted_price": float("nan"), "current_price": 200.0}
outcome = execute_trade_decision("TESTX", nan_ensemble, mode="paper", verbose=False)
assert outcome["executed"] is False
assert outcome["signal"] is None
logged = trade_log.read_all_trades()
assert logged[-1]["status"] == "ERROR"
print("Test 23b (trade_executor: NaN prediction blocked cleanly + logged, not an uncaught crash) passed")

# BUY signal, paper mode -> should execute against the paper broker
config.PAPER_STARTING_CASH = 100000
buy_ensemble = {"predicted_pct_change": 2.0, "final_predicted_price": 204.0, "current_price": 200.0}
outcome = execute_trade_decision("TESTX", buy_ensemble, mode="paper", verbose=False)
assert outcome["executed"] is True, outcome
assert outcome["signal"]["action"] == "BUY"
print("Test 24 (trade_executor: BUY signal executes against paper broker) passed:", outcome["quantity"])

# live mode without ENABLE_LIVE_TRADING -- must be blocked, never reach a real broker
original_flag = config.ENABLE_LIVE_TRADING
config.ENABLE_LIVE_TRADING = False
outcome = execute_trade_decision("TESTX", buy_ensemble, mode="live", broker_name="upstox", verbose=False)
assert outcome["executed"] is False
assert "disabled" in outcome["reason"].lower()
print("Test 25 (trade_executor: live mode blocked without ENABLE_LIVE_TRADING) passed")
config.ENABLE_LIVE_TRADING = original_flag

# quantity_override lets a caller force an exact size, still validated by risk_manager
outcome = execute_trade_decision("TESTX", buy_ensemble, mode="paper", quantity_override=5, verbose=False)
assert outcome["executed"] is True
assert outcome["quantity"] == 5
print("Test 26 (trade_executor: quantity_override respected and still validated) passed")

# quantity_override that violates risk limits should be BLOCKED, not silently executed
outcome = execute_trade_decision("TESTX", buy_ensemble, mode="paper", quantity_override=999999, verbose=False)
assert outcome["executed"] is False
print("Test 27 (trade_executor: oversized quantity_override blocked by risk_manager) passed")

# regression test: mode="paper" with a mismatched broker_name must still use
# PaperBroker AND log broker="paper" (not the mismatched name) -- guards
# against a misleading audit trail
outcome = execute_trade_decision("TESTX", buy_ensemble, mode="paper", broker_name="upstox", verbose=False)
assert outcome["executed"] is True
logged_rows = trade_log.read_all_trades()
last_row = logged_rows[-1]
assert last_row["mode"] == "paper"
assert last_row["broker"] == "paper", last_row  # NOT "upstox", even though that was passed in
print("Test 28 (trade_executor: paper mode always logs broker=paper, never a mismatched name) passed")

# ---------------------------------------------------------------------------
# trade_executor: full BUY -> SELL lifecycle (regression test for the
# capital-vs-holdings sizing bug). A SELL signal on a ticker with NO position
# must be blocked, not phantom-executed; after a BUY establishes a position,
# a SELL signal must then size itself off that position, not off cash.
# ---------------------------------------------------------------------------
fresh_broker_path = os.path.join(TMP_DIR, "lifecycle_portfolio.json")
config.PAPER_PORTFOLIO_PATH = fresh_broker_path
config.TRADE_LOG_PATH = os.path.join(TMP_DIR, "lifecycle_trade_log.csv")  # isolated from earlier tests' trade counts
config.PAPER_STARTING_CASH = 100000

sell_ensemble = {"predicted_pct_change": -2.0, "final_predicted_price": 196.0, "current_price": 200.0}
outcome = execute_trade_decision("LIFECYCLE", sell_ensemble, mode="paper", verbose=False)
assert outcome["executed"] is False
assert "no position" in outcome["reason"].lower()
print("Test 29 (trade_executor: SELL with zero position correctly blocked) passed")

buy_ensemble2 = {"predicted_pct_change": 2.0, "final_predicted_price": 204.0, "current_price": 200.0}
outcome = execute_trade_decision("LIFECYCLE", buy_ensemble2, mode="paper", verbose=False)
assert outcome["executed"] is True
bought_qty = outcome["quantity"]
print(f"Test 30 (trade_executor: BUY establishes a position of {bought_qty} shares) passed")

outcome = execute_trade_decision("LIFECYCLE", sell_ensemble, mode="paper", verbose=False)
assert outcome["executed"] is True, outcome
assert outcome["quantity"] == bought_qty, (outcome["quantity"], bought_qty)  # sells the FULL position
print(f"Test 31 (trade_executor: SELL after a BUY sizes off the held position ({outcome['quantity']} "
      f"shares), not off cash) passed")

# ---------------------------------------------------------------------------
# trade_executor: broker-level exceptions must never crash the run and must
# always still produce an audit log entry (regression test for a real gap
# found during QA: get_available_capital()/get_position()/place_order()
# raising was previously uncaught in places, which could crash trade.py
# with zero record that an order was even attempted)
# ---------------------------------------------------------------------------
from unittest.mock import patch, MagicMock

config.TRADE_LOG_PATH = os.path.join(TMP_DIR, "exception_trade_log.csv")
config.ENABLE_LIVE_TRADING = True

broken_broker = MagicMock()
broken_broker.get_available_capital.side_effect = RuntimeError("Upstox funds request failed (network error)")
broken_broker.get_position.return_value = 0

with patch("trading.trade_executor._get_broker", return_value=broken_broker):
    outcome = execute_trade_decision("BROKENX", buy_ensemble, mode="live", broker_name="upstox",
                                      auto_confirm=True, verbose=False)
    assert outcome["executed"] is False
    assert "network error" in outcome["reason"].lower() or "funds request failed" in outcome["reason"].lower()
    logged = trade_log.read_all_trades()
    assert logged[-1]["status"] == "BLOCKED"
    print("Test 32 (trade_executor: get_available_capital() failure blocked cleanly + logged, "
          "not an uncaught crash) passed")

broken_broker2 = MagicMock()
broken_broker2.get_available_capital.return_value = 100000
broken_broker2.get_position.return_value = 0
broken_broker2.place_order.side_effect = RuntimeError("Connection reset by peer")

with patch("trading.trade_executor._get_broker", return_value=broken_broker2):
    outcome = execute_trade_decision("BROKENY", buy_ensemble, mode="live", broker_name="upstox",
                                      auto_confirm=True, verbose=False)
    assert outcome["executed"] is False
    logged = trade_log.read_all_trades()
    assert logged[-1]["status"] == "ERROR"
    assert logged[-1]["ticker"] == "BROKENY"
    print("Test 33 (trade_executor: place_order() crash still produces an audit log entry, "
          "guaranteeing no trade attempt goes unrecorded) passed")

config.ENABLE_LIVE_TRADING = False

shutil.rmtree(TMP_DIR, ignore_errors=True)
print("\nALL TRADING TESTS PASSED")
