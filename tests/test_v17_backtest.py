import numpy as np
import pytest

import backtest as bt

D = [f"2026-03-0{i}" for i in range(2, 9)]
CLOSE = [100, 101, 102, 100, 99, 103, 104.0]
OPEN = [100, 100.5, 101.5, 101, 99.5, 100, 103.5]
PRED = [100, 102, 103, 99, 98, 104, 105.0]


def run(**kw):
    base = dict(initial_capital=1000, transaction_cost_pct=0, buy_threshold_pct=0.5, sell_threshold_pct=-0.5)
    base.update(kw)
    return bt.run_backtest(D, CLOSE, PRED, open_prices=OPEN, **base)


def test_default_mode_is_backward_compatible_close_fill():
    r = run()
    assert r.execution == "close" and r.trades[0].price == 101.0          # day-1 close


def test_next_open_fills_at_the_open_not_the_close():
    r = run(execution="next_open")
    assert [t.price for t in r.trades][:2] == [100.5, 101.0]
    assert "OPEN" in r.assumptions


def test_signal_cannot_depend_on_the_fill_day_close():
    """Changing day-i close (info not available at decision time) must not change day-i's decision or fill."""
    a = run(execution="next_open")
    close2 = list(CLOSE); close2[1] = 150.0                              # future of the day-0 decision
    b = bt.run_backtest(D, close2, PRED, open_prices=OPEN, initial_capital=1000, transaction_cost_pct=0,
                        buy_threshold_pct=0.5, sell_threshold_pct=-0.5, execution="next_open")
    assert a.trades[0].date == b.trades[0].date and a.trades[0].price == b.trades[0].price


def test_missing_open_means_no_fill_and_is_counted_never_guessed():
    opens = list(OPEN); opens[1] = float("nan")
    r = bt.run_backtest(D, CLOSE, PRED, open_prices=opens, initial_capital=1000, transaction_cost_pct=0,
                        buy_threshold_pct=0.5, sell_threshold_pct=-0.5, execution="next_open")
    assert r.skipped_fills >= 1 and all(t.date != D[1] for t in r.trades)


def test_next_open_requires_open_prices():
    with pytest.raises(ValueError):
        bt.run_backtest(D, CLOSE, PRED, execution="next_open")
    with pytest.raises(ValueError):
        run(execution="whenever")


def test_slippage_hurts_buys_and_sells_and_costs_are_charged():
    free = run(execution="next_open", slippage_bps=0, transaction_cost_pct=0)
    slip = run(execution="next_open", slippage_bps=50, transaction_cost_pct=0)
    costly = run(execution="next_open", slippage_bps=0, transaction_cost_pct=0.5)
    assert slip.trades[0].price > free.trades[0].price and slip.total_return_pct < free.total_return_pct
    assert costly.total_return_pct < free.total_return_pct and costly.total_transaction_costs > 0


def test_buy_and_hold_benchmark_uses_the_same_entry_rule_and_costs():
    r = run(execution="next_open", transaction_cost_pct=0.1, slippage_bps=10)
    shares = 1000 * (1 - 0.001) / (OPEN[1] * 1.001)
    expected_final = shares * CLOSE[-1] * (1 - 0.001)
    assert r.buy_hold_return_pct == pytest.approx((expected_final / 1000 - 1) * 100)
    assert len(r.benchmark_curve) == len(D) and r.benchmark_curve[0]["portfolio_value"] == 1000


def test_drawdown_curve_matches_reported_max_drawdown():
    r = run(execution="next_open")
    worst = min(d["drawdown_pct"] for d in r.drawdown_curve)
    assert worst == pytest.approx(r.max_drawdown_pct, abs=1e-9) and worst <= 0


def test_equity_is_marked_to_close_not_to_fill_price():
    r = run(execution="next_open")
    # after BUY on day 1 at open 100.5, equity at that day's close 101 must reflect the close
    shares = r.trades[0].shares
    cash = 1000 - shares * 100.5
    assert r.equity_curve[1]["portfolio_value"] == pytest.approx(cash + shares * CLOSE[1])


def test_position_never_exceeds_capital_long_only():
    r = run(execution="next_open")
    for e in r.equity_curve:
        assert np.isfinite(e["portfolio_value"]) and e["portfolio_value"] > 0
    assert all(t.shares >= 0 for t in r.trades)
