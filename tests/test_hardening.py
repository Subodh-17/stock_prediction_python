"""
tests/test_hardening.py
------------------------
Regression tests for the release-hardening pass. ADDITIVE -- no existing
test was removed, weakened, or skipped.

Covers:
  1. Secret redaction across every credential this project holds
  2. Broker failure handling (network, HTTP error, malformed body) -- mocked,
     no real order is ever placed
  3. Live-trading security: disabled by default, invalid broker, excessive
     position, daily trade limit, duplicate orders
  4. Numerical robustness: NaN/inf/zero-division/empty/constant/duplicate/
     unsorted/timezone-aware inputs
  5. Checkpoint architecture validation
  6. Request-size limit and security headers (the whole upload surface --
     this app has no file-upload routes)
"""
import os
import sys
import json
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import requests

import config
import data_loader
import metrics_utils
import security
from data_sources.base import DataSourceError


# ---------------------------------------------------------------------------
# 1. Secret redaction
# ---------------------------------------------------------------------------
SECRET_FIELDS = [
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY",
    "DEEPSEEK_API_KEY", "UPSTOX_CLIENT_ID", "UPSTOX_CLIENT_SECRET",
    "GROWW_ACCESS_TOKEN", "GROWW_API_KEY", "GROWW_API_SECRET",
    "WEBAPP_LIVE_TRADING_TOKEN",
]


@pytest.mark.parametrize("field", SECRET_FIELDS)
def test_every_configured_secret_is_redacted(field, monkeypatch):
    """
    Redaction used to cover only the five LLM keys. Broker and web-app
    credentials were not covered at all, and those are exactly the paths that
    embedded raw upstream response bodies.
    """
    sentinel = f"SENTINEL-{field}-abcdef1234567890"
    monkeypatch.setattr(config, field, sentinel, raising=False)
    out = security.redact(f"request failed, value was {sentinel} -- end")
    assert sentinel not in out
    assert security.REDACTED in out


def test_redaction_catches_credentials_never_in_our_config():
    """
    Pattern-based scrubbing, so a credential belonging to the UPSTREAM
    service (echoed back in its own error body) is caught too.
    """
    for probe in (
        '{"authorization":"Bearer abcdefghijklmnopqrstuvwxyz"}',
        "https://api.example.com/v1?key=AIzaSyAbCdEfGhIjKlMnOpQrStUvWx1234",
        "auth failed for api_key=zzzzzzzzzzzzzzzzzzzz",
        "openai said sk-ant-abcdefghijklmnopqrst was invalid",
    ):
        out = security.redact(probe)
        assert security.REDACTED in out, probe


def test_short_values_are_not_redacted_into_nonsense(monkeypatch):
    """A 3-char placeholder must not blank out unrelated text."""
    monkeypatch.setattr(config, "GROWW_API_KEY", "abc", raising=False)
    assert security.redact("the abacus was abc-shaped") == "the abacus was abc-shaped"


def test_upstream_bodies_are_truncated_not_dumped():
    out = security.safe_upstream_detail("A" * 10_000, 500)
    assert len(out) < security.MAX_UPSTREAM_DETAIL_CHARS + 60
    assert "truncated" in out
    assert security.safe_upstream_detail("", 502) == "HTTP 502: (no response body)"


def test_token_endpoint_bodies_are_never_echoed(monkeypatch):
    """
    VERIFIED LEAK, now a regression test: a token returned under an
    unexpected key was echoed verbatim out of the Groww token endpoint.
    """
    from data_sources import groww_source

    monkeypatch.setattr(config, "GROWW_ACCESS_TOKEN", "", raising=False)
    monkeypatch.setattr(config, "GROWW_API_KEY", "key-aaaaaaaaaaaaaaaa", raising=False)
    monkeypatch.setattr(config, "GROWW_API_SECRET", "secret-bbbbbbbbbbbb", raising=False)

    live_token = "LIVE-TOKEN-MUST-NOT-APPEAR-9f8a7b6c"

    class _Resp:
        status_code = 200
        text = '{"accessToken":"%s"}' % live_token

        def json(self):
            return {"accessToken": live_token}

    with patch.object(groww_source.requests, "post", return_value=_Resp()):
        with pytest.raises(DataSourceError) as exc:
            groww_source.get_access_token()
    assert live_token not in str(exc.value)
    assert "accessToken" not in str(exc.value)


def test_logging_filter_redacts_secrets(monkeypatch, caplog):
    import logging_utils

    monkeypatch.setattr(config, "GROWW_API_SECRET", "supersecret-value-123456", raising=False)
    log = logging_utils.get_logger("test")
    with caplog.at_level("WARNING", logger="stockpred.test"):
        log.warning("failed with %s", "supersecret-value-123456")
    assert "supersecret-value-123456" not in caplog.text


# ---------------------------------------------------------------------------
# 2. Broker failure handling (all mocked -- no real order is ever placed)
# ---------------------------------------------------------------------------
def _broker(module_name):
    from trading.brokers import groww_broker, upstox_broker
    mod = {"groww": groww_broker, "upstox": upstox_broker}[module_name]
    cls = {"groww": groww_broker.GrowwBroker, "upstox": upstox_broker.UpstoxBroker}[module_name]
    broker = cls.__new__(cls)          # bypass __init__, which would authenticate
    broker.access_token = "test-token-not-real"
    return mod, broker


@pytest.mark.parametrize("which", ["groww", "upstox"])
def test_broker_network_failure_returns_error_not_exception(which):
    mod, broker = _broker(which)
    with patch.object(mod.requests, "post",
                      side_effect=requests.exceptions.ConnectionError("no route to host")):
        if which == "upstox":
            with patch.object(mod, "_resolve_instrument_key", return_value="KEY"):
                result = broker.place_order("RELIANCE", "BUY", 1, price=100.0)
        else:
            result = broker.place_order("RELIANCE", "BUY", 1, price=100.0)
    assert result["status"] == "ERROR"
    assert result["broker_order_id"] == ""
    # The operator must be told the order's fate is genuinely unknown.
    assert "may or may not" in result["message"]


@pytest.mark.parametrize("which", ["groww", "upstox"])
def test_broker_http_error_never_echoes_credentials(which, monkeypatch):
    monkeypatch.setattr(config, "GROWW_API_KEY", "groww-key-aaaaaaaaaaaa", raising=False)
    mod, broker = _broker(which)

    class _Resp:
        status_code = 403
        text = '{"echoed_authorization":"Bearer groww-key-aaaaaaaaaaaa"}'

        def json(self):
            return {}

    with patch.object(mod.requests, "post", return_value=_Resp()):
        if which == "upstox":
            with patch.object(mod, "_resolve_instrument_key", return_value="KEY"):
                result = broker.place_order("RELIANCE", "BUY", 1, price=100.0)
        else:
            result = broker.place_order("RELIANCE", "BUY", 1, price=100.0)
    assert result["status"] == "ERROR"
    assert "groww-key-aaaaaaaaaaaa" not in result["message"]


@pytest.mark.parametrize("which", ["groww", "upstox"])
def test_broker_rejects_unknown_action_without_any_network_call(which):
    mod, broker = _broker(which)
    with patch.object(mod.requests, "post", side_effect=AssertionError("must not call the network")):
        result = broker.place_order("RELIANCE", "SIDEWAYS", 1, price=100.0)
    assert result["status"] == "REJECTED"


# ---------------------------------------------------------------------------
# 3. Live-trading security
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_trading(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRADE_LOG_PATH", str(tmp_path / "trades.csv"))
    monkeypatch.setattr(config, "PAPER_PORTFOLIO_PATH", str(tmp_path / "portfolio.json"))
    monkeypatch.setattr(config, "PAPER_STARTING_CASH", 100_000.0)
    return tmp_path


BUY_SIGNAL = {
    "final_predicted_price": 110.0, "current_price": 100.0, "predicted_pct_change": 10.0,
    "component_predictions": {"linear_regression": 110.0, "lstm": 110.0},
    "weights_used": {"linear_regression": 0.5, "lstm": 0.5, "llm_panel": 0.0},
}


def test_live_trading_is_disabled_on_a_fresh_install(isolated_trading, monkeypatch):
    """A fresh install must not be able to reach a real broker."""
    monkeypatch.setattr(config, "ENABLE_LIVE_TRADING", False)
    from trading.trade_executor import execute_trade_decision
    from trading.brokers import upstox_broker

    with patch.object(upstox_broker.UpstoxBroker, "place_order",
                      side_effect=AssertionError("a real order was attempted")):
        out = execute_trade_decision("AAA", BUY_SIGNAL, mode="live", broker_name="upstox",
                                      auto_confirm=True, verbose=False)
    assert out["executed"] is False
    assert "Live trading is disabled" in out["reason"]


def test_live_trading_rejects_an_invalid_broker(isolated_trading, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_LIVE_TRADING", True)
    from trading.trade_executor import execute_trade_decision

    out = execute_trade_decision("AAA", BUY_SIGNAL, mode="live", broker_name="not-a-broker",
                                  auto_confirm=True, verbose=False)
    assert out["executed"] is False
    assert "Unknown live broker" in out["reason"]


def test_excessive_position_is_blocked(isolated_trading, monkeypatch):
    """A hand-picked quantity that skips sizing is still caught by validate_order."""
    monkeypatch.setattr(config, "MAX_ORDER_VALUE", 1_000.0)
    monkeypatch.setattr(config, "MAX_POSITION_PCT", 0.10)
    from trading.trade_executor import execute_trade_decision

    out = execute_trade_decision("AAA", BUY_SIGNAL, mode="paper", broker_name="paper",
                                  quantity_override=10_000, auto_confirm=True, verbose=False)
    assert out["executed"] is False
    assert "exceeds" in out["reason"]


def test_daily_trade_limit_blocks_further_orders(isolated_trading, monkeypatch):
    monkeypatch.setattr(config, "MAX_TRADES_PER_DAY", 1)
    from trading import trade_log
    from trading.trade_executor import execute_trade_decision

    trade_log.log_trade("AAA", "BUY", "paper", "paper", 1.0, 100.0, 110.0,
                        quantity=1, order_value=100.0, status="FILLED", message="seed")
    out = execute_trade_decision("AAA", BUY_SIGNAL, mode="paper", broker_name="paper",
                                  auto_confirm=True, verbose=False)
    assert out["executed"] is False
    assert "Daily trade limit" in out["reason"]


def test_duplicate_live_order_is_blocked(isolated_trading, monkeypatch):
    monkeypatch.setattr(config, "ENABLE_LIVE_TRADING", True)
    monkeypatch.setattr(config, "DUPLICATE_ORDER_WINDOW_SECONDS", 300)
    monkeypatch.setattr(config, "MAX_TRADES_PER_DAY", 50)
    from trading import trade_log
    from trading.trade_executor import execute_trade_decision
    from trading.brokers import upstox_broker

    qty = 10
    trade_log.log_trade("AAA", "BUY", "live", "upstox", 10.0, 100.0, 110.0,
                        quantity=qty, order_value=qty * 100.0, status="OPEN", message="first")

    with patch.object(upstox_broker.UpstoxBroker, "__init__", return_value=None), \
         patch.object(upstox_broker.UpstoxBroker, "get_available_capital", return_value=1_000_000.0), \
         patch.object(upstox_broker.UpstoxBroker, "get_position", return_value=0.0), \
         patch.object(upstox_broker.UpstoxBroker, "place_order",
                      side_effect=AssertionError("duplicate order reached the broker")):
        out = execute_trade_decision("AAA", BUY_SIGNAL, mode="live", broker_name="upstox",
                                      quantity_override=qty, auto_confirm=True, verbose=False)
    assert out["executed"] is False
    assert "Duplicate order blocked" in out["reason"]


def test_paper_mode_never_reaches_a_live_broker(isolated_trading):
    from trading.trade_executor import execute_trade_decision
    from trading.brokers import upstox_broker, groww_broker

    with patch.object(upstox_broker.UpstoxBroker, "place_order",
                      side_effect=AssertionError("live broker used in paper mode")), \
         patch.object(groww_broker.GrowwBroker, "place_order",
                      side_effect=AssertionError("live broker used in paper mode")):
        out = execute_trade_decision("AAA", BUY_SIGNAL, mode="paper", broker_name="upstox",
                                      auto_confirm=True, verbose=False)
    assert out["executed"] is True
    assert "[PAPER]" in out["reason"]


# ---------------------------------------------------------------------------
# 4. Numerical robustness
# ---------------------------------------------------------------------------
def test_mape_does_not_become_infinite_on_a_zero_price():
    """Plain MAPE divides by the actual value; one zero made the metric +inf."""
    value = metrics_utils.safe_mape([0.0, 10.0, 20.0], [1.0, 11.0, 19.0])
    assert np.isfinite(value)


def test_mape_is_nan_when_nothing_is_measurable():
    assert np.isnan(metrics_utils.safe_mape([0.0, 0.0], [1.0, 2.0]))


def test_directional_accuracy_is_nan_for_too_short_a_series():
    assert np.isnan(metrics_utils.directional_accuracy([1.0], [1.0]))
    assert np.isnan(metrics_utils.directional_accuracy([1.0, 2.0], [1.0]))


def test_empty_and_tiny_dataframes_raise_clearly_rather_than_crashing():
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    report = data_loader.check_data_quality(empty, "EMPTY")
    assert not report.is_valid

    tiny = pd.DataFrame(
        {"Open": [1.0] * 5, "High": [1.0] * 5, "Low": [1.0] * 5,
         "Close": [1.0] * 5, "Volume": [1] * 5},
        index=pd.bdate_range("2024-01-01", periods=5),
    )
    with pytest.raises(data_loader.InsufficientDataError):
        data_loader.validate_sufficient_data(tiny)


def _flat(n=80, price=100.0):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"Open": [price] * n, "High": [price] * n, "Low": [price] * n,
         "Close": [price] * n, "Volume": [1000] * n},
        index=idx,
    )


def test_constant_price_series_produces_finite_metrics():
    from models.linear_regression_model import LinearRegressionPredictor

    X_tr, X_te, y_tr, y_te, cols, _ = data_loader.prepare_linear_regression_data(_flat())
    ev = LinearRegressionPredictor().train(X_tr, y_tr, cols).evaluate(X_te, y_te)
    for key in ("rmse", "mae", "r2", "mape"):
        assert np.isfinite(ev[key]), f"{key} was not finite on a constant series"


def test_missing_required_columns_are_reported_not_ignored():
    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    report = data_loader.check_data_quality(df, "PARTIAL")
    assert not report.is_valid
    assert any("Missing required column" in i for i in report.issues)


def test_duplicate_and_unsorted_timestamps_are_detected():
    df = _flat(n=10)
    dup = df.copy()
    dup.index = list(df.index[:-1]) + [df.index[-2]]
    assert any("duplicate" in i.lower() for i in data_loader.check_data_quality(dup, "D").issues)
    assert any("chronological" in i.lower()
               for i in data_loader.check_data_quality(df.iloc[::-1], "U").issues)


def test_timezone_aware_index_is_handled():
    df = _flat()
    df.index = df.index.tz_localize("UTC")
    _, _, _, y_test, _, dates = data_loader.prepare_linear_regression_data(df)
    assert len(dates) == len(y_test) > 0


def test_negative_and_zero_prices_are_flagged_as_issues():
    df = _flat(n=20)
    df.iloc[5, df.columns.get_loc("Close")] = 0.0
    df.iloc[6, df.columns.get_loc("Open")] = -1.0
    issues = " ".join(data_loader.check_data_quality(df, "BAD").issues).lower()
    assert "zero or negative" in issues


def test_data_quality_never_fabricates_or_mutates_data():
    df = _flat(n=20)
    before = df.copy(deep=True)
    data_loader.check_data_quality(df, "X")
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# 5. Checkpoint architecture validation
# ---------------------------------------------------------------------------
def test_checkpoint_with_mismatched_weights_is_refused(tmp_path, monkeypatch):
    """
    The metadata checks compare a JSON sidecar against the run. This verifies
    the loaded MODEL itself, so a swapped model.keras cannot slip through.
    """
    import checkpoints as ckpt_mod
    from models.lstm_model import LSTMPredictor

    ckpt_dir = str(tmp_path / "ckpts")
    monkeypatch.setattr(config, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(ckpt_mod.config, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(config, "TRAIN_TEST_SPLIT", 0.8)
    monkeypatch.setattr(ckpt_mod.config, "TRAIN_TEST_SPLIT", 0.8)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler().fit(np.linspace(0, 1, 50).reshape(-1, 1))

    model = LSTMPredictor(lookback=10)
    ckpt_mod.save_checkpoint("arch", model, scaler, ticker="TESTX", period="1y",
                              source="synthetic", n_train_rows=50, target_mode="return")

    # Swap in weights built for a different sequence length, leaving the
    # metadata untouched -- exactly the desync this check exists to catch.
    other = LSTMPredictor(lookback=25)
    other.model.save(os.path.join(ckpt_dir, "arch", "model.keras"))

    with pytest.raises(ckpt_mod.CheckpointMismatchError) as exc:
        ckpt_mod.load_checkpoint("arch", ticker="TESTX", lookback=10,
                                  target_mode="return", train_test_split=0.8)
    assert "input shape" in str(exc.value)


def test_checkpoint_with_an_unfitted_scaler_is_refused(tmp_path, monkeypatch):
    """
    save_checkpoint() itself now refuses to write an unfitted scaler at all
    (_scaler_to_dict raises immediately -- see
    test_saving_an_unfitted_scaler_is_refused_outright below), so to exercise
    load_checkpoint()'s own defense-in-depth check, this simulates a
    scaler.json that was corrupted or hand-edited after the fact (plain JSON
    is directly editable, unlike a pickle blob) to no longer record
    n_features_in_ -- exactly what an unfitted scaler looks like.
    """
    import json
    import checkpoints as ckpt_mod
    from models.lstm_model import LSTMPredictor
    from sklearn.preprocessing import MinMaxScaler

    ckpt_dir = str(tmp_path / "ckpts")
    monkeypatch.setattr(config, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(ckpt_mod.config, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(config, "TRAIN_TEST_SPLIT", 0.8)
    monkeypatch.setattr(ckpt_mod.config, "TRAIN_TEST_SPLIT", 0.8)

    fitted = MinMaxScaler().fit(np.linspace(0, 1, 50).reshape(-1, 1))
    ckpt_mod.save_checkpoint("noscale", LSTMPredictor(lookback=10), fitted, ticker="TESTX",
                              period="1y", source="synthetic", n_train_rows=50,
                              target_mode="return")
    corrupted = {
        "feature_range": [0, 1], "scale_": [1.0], "min_": [0.0],
        "data_min_": [0.0], "data_max_": [1.0], "data_range_": [1.0],
        "n_features_in_": None, "n_samples_seen_": None, "copy": True, "clip": False,
    }
    with open(os.path.join(ckpt_dir, "noscale", "scaler.json"), "w") as f:
        json.dump(corrupted, f)

    with pytest.raises(ckpt_mod.CheckpointMismatchError) as exc:
        ckpt_mod.load_checkpoint("noscale", ticker="TESTX", lookback=10,
                                  target_mode="return", train_test_split=0.8)
    assert "never fitted" in str(exc.value)


def test_saving_an_unfitted_scaler_is_refused_outright(tmp_path, monkeypatch):
    """save_checkpoint() should never even be able to write an unfitted
    scaler in the first place -- the corrupted-file scenario above is for a
    file that was tampered with after a valid save, not the normal path."""
    import checkpoints as ckpt_mod
    from models.lstm_model import LSTMPredictor
    from sklearn.preprocessing import MinMaxScaler

    ckpt_dir = str(tmp_path / "ckpts")
    monkeypatch.setattr(config, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(ckpt_mod.config, "CHECKPOINT_DIR", ckpt_dir)

    with pytest.raises(ValueError, match="unfitted"):
        ckpt_mod.save_checkpoint("neverfit", LSTMPredictor(lookback=10), MinMaxScaler(),
                                  ticker="TESTX", period="1y", source="synthetic",
                                  n_train_rows=50, target_mode="return")


def test_scaler_json_roundtrip_matches_original_transform():
    """The whole point of the JSON scaler format: it must reproduce the
    original fitted scaler's transform/inverse_transform exactly, not just
    exist. Directly exercises the pickle-replacement (see checkpoints.py's
    module docstring, SECURITY section) rather than assuming it works."""
    import checkpoints as ckpt_mod
    from sklearn.preprocessing import MinMaxScaler

    original = MinMaxScaler(feature_range=(0, 1)).fit(
        np.array([[1.0], [2.0], [3.5], [7.25]])
    )
    restored = ckpt_mod._scaler_from_dict(ckpt_mod._scaler_to_dict(original))

    sample = np.array([[0.5], [2.5], [10.0]])
    np.testing.assert_allclose(original.transform(sample), restored.transform(sample))
    transformed = original.transform(sample)
    np.testing.assert_allclose(original.inverse_transform(transformed),
                                restored.inverse_transform(transformed))


def test_scaler_json_contains_no_pickle_opcodes_on_disk(tmp_path, monkeypatch):
    """Regression test for the pickle removal itself: a saved checkpoint's
    scaler file must be plain, human-readable JSON -- not a pickle stream --
    so there is no deserialization gadget on disk to exploit even if the
    checkpoint directory were ever compromised."""
    import checkpoints as ckpt_mod
    from models.lstm_model import LSTMPredictor
    from sklearn.preprocessing import MinMaxScaler

    ckpt_dir = str(tmp_path / "ckpts")
    monkeypatch.setattr(config, "CHECKPOINT_DIR", ckpt_dir)
    monkeypatch.setattr(ckpt_mod.config, "CHECKPOINT_DIR", ckpt_dir)

    fitted = MinMaxScaler().fit(np.linspace(0, 1, 50).reshape(-1, 1))
    ckpt_mod.save_checkpoint("plaintext", LSTMPredictor(lookback=10), fitted, ticker="TESTX",
                              period="1y", source="synthetic", n_train_rows=50,
                              target_mode="return")

    scaler_path = os.path.join(ckpt_dir, "plaintext", "scaler.json")
    assert not os.path.exists(os.path.join(ckpt_dir, "plaintext", "scaler.pkl"))
    with open(scaler_path, "rb") as f:
        raw = f.read()
    # A pickle stream starts with a PROTO opcode (b'\x80') for protocol >= 2;
    # valid JSON never starts with that byte. Also just parse it as JSON
    # outright -- pickle.dumps output is never valid JSON.
    assert not raw.startswith(b"\x80")
    json.loads(raw.decode("utf-8"))  # raises if this is anything but plain JSON



def test_checkpoint_name_cannot_traverse_paths():
    import checkpoints as ckpt_mod
    for bad in ("../evil", "a/b", "..", "x\\y"):
        with pytest.raises(ValueError):
            ckpt_mod._checkpoint_dir(bad)


# ---------------------------------------------------------------------------
# 6. Web surface: request-size limit and security headers
# ---------------------------------------------------------------------------
def _client():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "webapp"))
    import app as webapp_module
    webapp_module.app.config["TESTING"] = True
    return webapp_module.app.test_client()


def test_security_headers_are_present():
    resp = _client().get("/")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]
    assert resp.headers["Cache-Control"] == "no-store"


def test_oversized_request_is_rejected():
    resp = _client().post("/predict", data={"ticker": "A" * (512 * 1024)})
    assert resp.status_code == 413
    assert b"Traceback" not in resp.data


def test_app_exposes_no_file_upload_routes():
    """
    The whole file-upload attack surface: there isn't one. Asserted rather
    than assumed, so adding an upload route later trips this test and forces
    the extension/size/traversal handling Phase 13 asks for.
    """
    source = open(os.path.join(os.path.dirname(__file__), "..", "webapp", "app.py")).read()
    assert "request.files" not in source
    assert "send_file" not in source


def test_health_endpoint_needs_no_auth_and_leaks_nothing():
    resp = _client().get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert set(body) <= {"status", "service"}


def test_malformed_json_is_a_clean_400_not_a_traceback():
    resp = _client().post("/api/predict", data="{not json",
                           content_type="application/json")
    assert resp.status_code == 400
    assert b"Traceback" not in resp.data


# ---------------------------------------------------------------------------
# 7. CSRF protection on state-changing browser routes
# ---------------------------------------------------------------------------
def _csrf_client():
    c = _client()
    c.get("/")  # mints a session-bound csrf_token, the way a real browser would
    with c.session_transaction() as sess:
        token = sess["csrf_token"]
    return c, token


def test_post_without_csrf_token_is_rejected():
    c, _ = _csrf_client()
    resp = c.post("/predict", data={"ticker": "AAPL", "period": "2y", "epochs": "3"})
    assert resp.status_code == 403
    assert b"Traceback" not in resp.data


def test_post_with_invalid_csrf_token_is_rejected():
    c, _ = _csrf_client()
    resp = c.post("/predict", data={"ticker": "AAPL", "period": "2y", "epochs": "3",
                                     "csrf_token": "not-the-real-token"})
    assert resp.status_code == 403


def test_post_with_valid_csrf_token_passes_the_csrf_check():
    """
    Doesn't assert the eventual 200/redirect (that depends on the full
    pipeline running, covered elsewhere) -- only that a valid token clears
    the CSRF gate specifically, i.e. the response is NOT the 403 CSRF
    rejection, and the request reaches real input validation instead.
    """
    c, token = _csrf_client()
    resp = c.post("/predict", data={"ticker": "", "period": "2y", "epochs": "3",
                                     "csrf_token": token})
    # Empty ticker is rejected by /predict's own validation (400), which
    # only runs once the CSRF check has already passed -- so a 400 here
    # (not 403) proves the valid token was accepted.
    assert resp.status_code == 400
    assert b"ticker symbol" in resp.data.lower()


def test_get_requests_are_never_csrf_checked():
    c, _ = _csrf_client()
    for path in ("/", "/results", "/export/csv", "/export/pdf", "/trade/confirm"):
        resp = c.get(path, follow_redirects=False)
        assert resp.status_code != 403, f"GET {path} should never be blocked by CSRF"


def test_api_routes_are_exempt_from_csrf_by_design():
    """
    /api/* carries no session-based authentication or capability (see the
    comment above `_enforce_csrf` in app.py), so it is deliberately excluded
    from the token requirement -- verified here rather than just documented,
    so the exemption can't silently grow to cover a future stateful route.
    An empty payload fails fast in /api/predict's own validation (before the
    pipeline ever runs), which is enough to prove the request reached the
    route handler and got a structured JSON error -- not the app's HTML CSRF
    rejection page, which a missing/invalid token would otherwise produce.
    """
    resp = _client().post("/api/predict", json={})
    assert resp.status_code == 400
    assert resp.is_json
    assert resp.get_json()["error"] == "invalid_ticker"
