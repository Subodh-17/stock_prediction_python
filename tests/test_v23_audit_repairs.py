"""
tests/test_v23_audit_repairs.py
---------------------------------
Regression tests verifying verified engineering fixes for Quant Workstation V23:
  - DEF-V23-01: trade_log formatting resilience with None or non-numeric values
  - DEF-V23-02: _TICKER_OK regex and /predict validation for exchange-prefixed tickers (NSE:, BSE:)
  - DEF-V23-03: _portfolio_snapshot quote failure isolation in webapp/api_v2.py
  - DEF-V23-04: risk_size boundary handling (non-positive quote price and negative portfolio equity)
  - DEF-V23-05: run_backtest input validation (non-positive capital, NaN/negative prices)
  - DEF-V23-06: PaperBroker float-tolerance position liquidation
  - DEF-V23-07: paper_order outage-cached quote rejection
  - DEF-V23-08: package_release REQUIRED list coverage for V23 deliverables
"""
import csv
import json
import os
import re
import sys
import zipfile
from unittest.mock import MagicMock, patch

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "webapp"))
sys.path.insert(0, ROOT)

import backtest as bt
from trading import trade_log
from trading.brokers.paper_broker import PaperBroker
import market_data as md
import scripts.package_release as pk
import webapp.app as flask_app
import webapp.api_v2 as api_v2


# ============================================================================
# DEF-V23-01: trade_log None formatting resilience
# ============================================================================
def test_trade_log_safely_handles_none_and_non_numeric_prices(tmp_path):
    log_path = str(tmp_path / "trade_log.csv")
    # Calling log_trade with None numeric fields must not raise TypeError
    trade_log.log_trade(
        ticker="TEST", action="ERROR", mode="paper", broker="paper",
        predicted_pct_change=None, current_price=None, predicted_price=None,
        quantity=None, order_value=None, status="ERROR", message="Failure test",
        path=log_path,
    )
    assert os.path.exists(log_path)
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "TEST"
    assert r["action"] == "ERROR"
    assert r["predicted_pct_change"] == ""
    assert r["current_price"] == ""
    assert r["predicted_price"] == ""
    assert r["quantity"] == "0"
    assert r["order_value"] == ""


# ============================================================================
# DEF-V23-02: _TICKER_OK regex and /predict acceptance for exchange prefixes
# ============================================================================
def test_ticker_ok_regex_accepts_exchange_prefixed_tickers():
    assert flask_app._TICKER_OK.match("NSE:RELIANCE")
    assert flask_app._TICKER_OK.match("BSE:TCS")
    assert flask_app._TICKER_OK.match("AAPL")
    assert flask_app._TICKER_OK.match("VOD.L")
    assert flask_app._TICKER_OK.match("^NSEI")
    assert flask_app._TICKER_OK.match("^GSPC")

    with flask_app.app.app_context():
        ticker, err = flask_app._validate_ticker("NSE:RELIANCE")
        assert err is None
        assert ticker == "NSE:RELIANCE"

        ticker, err = flask_app._validate_ticker("BSE:TCS")
        assert err is None
        assert ticker == "BSE:TCS"

        # Invalid characters still rejected
        ticker, err = flask_app._validate_ticker("<script>alert(1)</script>")
        assert err is not None
        assert err[1] == 400


# ============================================================================
# DEF-V23-03: _portfolio_snapshot quote failure isolation
# ============================================================================
def test_portfolio_snapshot_resilient_to_failing_or_corrupt_holding(tmp_path):
    port_path = str(tmp_path / "paper_portfolio.json")
    with open(port_path, "w") as f:
        json.dump({"cash": 100000.0, "holdings": {"VALID": 10, "INVALID::CORRUPT": 5}, "avg_price": {"VALID": 50.0, "INVALID::CORRUPT": 20.0}}, f)

    with patch("config.PAPER_PORTFOLIO_PATH", port_path):
        with patch.object(md.MarketDataService, "get_quote") as mock_get_quote:
            def side_effect_quote(sym, force=False):
                if sym == "VALID":
                    return md.Quote(provider="yahoo", symbol="VALID", exchange="US/UNSPECIFIED", price=60.0, freshness="delayed")
                raise md.InvalidSymbol(f"Invalid symbol: {sym}")

            mock_get_quote.side_effect = side_effect_quote
            b, positions, equity = api_v2._portfolio_snapshot()

            assert len(positions) == 2
            valid_p = next(p for p in positions if p["symbol"] == "VALID")
            invalid_p = next(p for p in positions if p["symbol"] == "INVALID::CORRUPT")

            assert valid_p["price"] == 60.0
            assert valid_p["market_value"] == 600.0
            assert valid_p["quote_freshness"] == "delayed"

            assert invalid_p["price"] is None
            assert invalid_p["market_value"] is None
            assert invalid_p["quote_freshness"] == "unavailable"

            # Equity incorporates valid position (100,000 cash + 600 stock)
            assert equity == 100600.0


# ============================================================================
# DEF-V23-04: risk_size boundary handling
# ============================================================================
def test_risk_size_protects_against_zero_price_and_negative_equity():
    client = flask_app.app.test_client()

    with patch.object(md.MarketDataService, "get_quote") as mock_get_quote:
        # 1. Price is zero -> must return HTTP 409, not ZeroDivisionError
        mock_get_quote.return_value = md.Quote(provider="test", symbol="AAPL", exchange="US/UNSPECIFIED", price=0.0, freshness="unavailable")
        resp = client.get("/api/v2/risk/size?symbol=AAPL&risk_pct=1&stop_pct=5")
        assert resp.status_code == 409
        assert resp.get_json()["error"] == "no_quote"

        # 2. Portfolio equity is negative -> suggested_quantity must be >= 0
        mock_get_quote.return_value = md.Quote(provider="test", symbol="AAPL", exchange="US/UNSPECIFIED", price=100.0, freshness="delayed")
        with patch.object(api_v2, "_portfolio_snapshot", return_value=(MagicMock(get_available_capital=lambda: -500.0), [], -500.0)):
            resp = client.get("/api/v2/risk/size?symbol=AAPL&risk_pct=1&stop_pct=5")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["suggested_quantity"] == 0
            assert data["quantity_by_risk"] == 0


# ============================================================================
# DEF-V23-05: run_backtest input validation
# ============================================================================
def test_run_backtest_validates_initial_capital_and_prices():
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    valid_prices = [100.0, 102.0, 101.0]

    # Non-positive initial_capital
    with pytest.raises(ValueError, match="initial_capital must be positive"):
        bt.run_backtest(dates, valid_prices, valid_prices, initial_capital=0)

    with pytest.raises(ValueError, match="initial_capital must be positive"):
        bt.run_backtest(dates, valid_prices, valid_prices, initial_capital=-1000)

    # Non-finite or non-positive actual prices
    with pytest.raises(ValueError, match="actual_prices must all be finite and positive"):
        bt.run_backtest(dates, [100.0, float("nan"), 101.0], valid_prices)

    with pytest.raises(ValueError, match="actual_prices must all be finite and positive"):
        bt.run_backtest(dates, [100.0, 0.0, 101.0], valid_prices)

    # Non-finite or non-positive predicted prices
    with pytest.raises(ValueError, match="predicted_prices must all be finite and positive"):
        bt.run_backtest(dates, valid_prices, [100.0, float("inf"), 101.0])

    with pytest.raises(ValueError, match="predicted_prices must all be finite and positive"):
        bt.run_backtest(dates, valid_prices, [100.0, -50.0, 101.0])


# ============================================================================
# DEF-V23-06: PaperBroker float-tolerance position liquidation
# ============================================================================
def test_paper_broker_liquidates_position_cleanly_on_float_tolerance(tmp_path):
    port_path = str(tmp_path / "paper_portfolio.json")
    b = PaperBroker(portfolio_path=port_path, starting_cash=10000.0)
    b.state["holdings"]["AAPL"] = 10.000000000000002
    b.state["avg_price"]["AAPL"] = 150.0

    res = b.place_order("AAPL", "SELL", 10, price=155.0)
    assert res["status"] == "FILLED"
    assert "AAPL" not in b.state["holdings"]
    assert "AAPL" not in b.state["avg_price"]
    assert b.state["cash"] == 10000.0 + 10 * 155.0


# ============================================================================
# DEF-V23-07: paper_order outage-cached quote rejection
# ============================================================================
def test_paper_order_rejects_outage_cached_quote():
    client = flask_app.app.test_client()
    with client.session_transaction() as sess:
        sess["csrf_token"] = "fake_csrf_token_for_test_32_chars"

    outage_quote = md.Quote(
        provider="yahoo", symbol="AAPL", exchange="US/UNSPECIFIED", price=150.0,
        freshness="delayed", quality_flags=["served_cached_during_outage"],
    )

    with patch.object(md.MarketDataService, "get_quote", return_value=outage_quote):
        resp = client.post(
            "/api/v2/paper/order",
            headers={"X-CSRF-Token": "fake_csrf_token_for_test_32_chars"},
            json={"symbol": "AAPL", "side": "BUY", "quantity": 1, "confirm": True},
        )
        assert resp.status_code == 409
        assert resp.get_json()["error"] == "no_fresh_quote"


# ============================================================================
# DEF-V23-08: package_release REQUIRED list coverage
# ============================================================================
def test_package_release_required_includes_v23_artifacts():
    v23_required = [
        "reports/V23_BASELINE_AUDIT.md",
        "reports/V23_DEFECT_REGISTER.md",
        "reports/V23_TEST_REPORT.md",
        "reports/V23_RUNTIME_ACCEPTANCE.md",
        "reports/V23_RELEASE_REPORT.md",
        "reports/V23_1_RELEASE_REPORT.md",
        "tests/test_v23_audit_repairs.py",
    ]
    for r in v23_required:
        assert r in pk.REQUIRED, f"V23 artifact {r} missing from package_release.py REQUIRED list"


# ============================================================================
# DEF-V23-10: package_release manifest exclusion and verification
# ============================================================================
def test_package_release_strictly_excludes_manifest_and_sha_files(tmp_path):
    src = tmp_path / "src"
    for r in pk.REQUIRED:
        p = src / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("ok\n")
    (src / ".env.example").write_text("FOO=bar\n")
    # Add junk manifests and sha files
    (src / "stale.manifest.json").write_text("{}")
    (src / "stale.sha256").write_text("0" * 64)
    (src / "nested" / "deep.manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (src / "nested" / "deep.manifest.json").write_text("{}")

    out = tmp_path / "test.zip"
    pk.package(str(src), str(out))
    assert out.exists()

    with zipfile.ZipFile(out, "r") as z:
        names = z.namelist()
        assert not any(n.endswith(".manifest.json") for n in names)
        assert not any(n.endswith(".sha256") for n in names)
        assert not any("stale" in n for n in names)

    # Verify external companion files were written
    assert (tmp_path / "test.zip.manifest.json").exists()
    assert (tmp_path / "test.zip.sha256").exists()
    assert pk.verify_archive(str(out)) == []
