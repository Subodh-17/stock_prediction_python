"""Regression: V19 (and the original engine) left the ENTRY fee out of trade-level P&L, so the ledger, win rate, profit factor and
average trade return were flattered. Trade P&L must reconcile with the equity curve. Synthetic paths, tests only."""
import numpy as np
import pytest

import backtest as bt

DATES = [f"2026-01-{i:02d}" for i in range(1, 6)]


def flat_round_trip(cost):
    # flat price; BUY on day 1, SELL on day 3
    return bt.run_backtest(DATES, [100.0] * 5, [100, 101, 101, 99, 99], initial_capital=1000, transaction_cost_pct=cost,
                           buy_threshold_pct=0.5, sell_threshold_pct=-0.5)


def test_trade_pnl_includes_the_entry_fee_and_equals_the_equity_change():
    r = flat_round_trip(1.0)
    sells = [t for t in r.trades if t.action == "SELL"]
    assert len(sells) == 1
    assert sells[0].pnl == pytest.approx(-19.9) and sells[0].pnl == pytest.approx(r.final_portfolio_value - 1000)
    assert r.total_transaction_costs == pytest.approx(19.9)
    assert r.avg_trade_return_pct == pytest.approx(-1.99, abs=0.01)                    # was -1.00 before the fix
    assert sells[0].cost_basis == pytest.approx(1000.0)                               # what the position actually cost, fee included


def test_a_gain_smaller_than_the_fees_is_a_loss_not_a_win():
    prices = [100.0, 100.0, 100.5, 100.5, 100.5]                                       # +0.5% held; fees 1% each way
    r = bt.run_backtest(DATES, prices, [100, 101, 101, 99, 99], initial_capital=1000, transaction_cost_pct=1.0,
                        buy_threshold_pct=0.5, sell_threshold_pct=-0.5)
    assert r.final_portfolio_value < 1000
    assert r.win_rate_pct == 0.0 and r.profit_factor == 0.0                            # before the fix this trade counted as a win


@pytest.mark.parametrize("execution", ["close", "next_open"])
@pytest.mark.parametrize("cost,slip", [(0.0, 0.0), (0.1, 0.0), (0.5, 5.0), (1.0, 25.0)])
def test_trade_level_pnl_reconciles_with_the_equity_curve_on_random_paths(execution, cost, slip):
    for seed in range(12):
        rng = np.random.default_rng(seed)
        n = 120
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
        opens = close * (1 + rng.normal(0, 0.002, n))
        pred = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.01, n))               # noisy "forecasts" -> many round trips
        dates = [f"d{i:03d}" for i in range(n)]
        r = bt.run_backtest(dates, close, pred, initial_capital=10_000, transaction_cost_pct=cost, buy_threshold_pct=0.3,
                            sell_threshold_pct=-0.3, open_prices=opens, execution=execution, slippage_bps=slip)
        closed = sum(t.pnl for t in r.trades if t.action == "SELL")
        if not r.open_position_at_end:
            assert closed == pytest.approx(r.final_portfolio_value - 10_000, abs=1e-6), (seed, execution, cost, slip)
        assert r.num_trades >= 0


def test_exposure_and_turnover_follow_their_definitions():
    r = flat_round_trip(1.0)
    # long at the close of day 1 and day 2 (sold on day 3 before the mark): 2 of 5 days
    assert r.exposure_pct == pytest.approx(40.0)
    traded = sum(t.shares * t.price + (t.cost if t.action == "BUY" else 0.0) for t in r.trades)
    assert r.turnover_x * 1000 == pytest.approx(traded)
    never = bt.run_backtest(DATES, [100.0] * 5, [100] * 5, initial_capital=1000, transaction_cost_pct=0.1)
    assert never.exposure_pct == 0.0 and never.turnover_x == 0.0 and never.num_trades == 0


def test_exposure_and_turnover_are_in_the_serialised_result_and_the_text_summary():
    r = flat_round_trip(0.1)
    d = r.to_dict()
    assert d["exposure_pct"] == pytest.approx(40.0) and "turnover_x" in d
    txt = r.summary_text()
    assert "Exposure" in txt and "of days in the market" in txt and "Turnover" in txt and "capital traded" in txt
