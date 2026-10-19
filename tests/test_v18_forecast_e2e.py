"""Runs the REAL forecast worker (real LR + LSTM, 1 epoch) on a synthetic fixture and checks the payload the UI renders.
Synthetic data lives only in this test; it is never a runtime fallback."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "webapp"), ROOT]

import config
import forecast_service as fs
from tests.test_v17_webapp import env, wait_done          # noqa: F401
from tests.v17_helpers import fixture_bars


@pytest.fixture
def real(env, monkeypatch, tmp_path):
    c, hdr, provider, svc = env
    import data_sources.yfinance_source as ys
    monkeypatch.setattr(ys, "fetch", lambda *a, **k: fixture_bars(400, seed=3, start="2024-01-02"))
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path / "ck"))
    monkeypatch.setattr(config, "LSTM_PREDICTION_TARGET", "return")
    monkeypatch.setattr(fs.market_data.config if hasattr(fs.market_data, "config") else config, "QUOTE_TTL_SECONDS", 15)
    return c, hdr, svc


def run_job(c, svc, epochs=1):
    job = fs.submit_forecast("AAPL", "2y", epochs, svc=svc)
    done = wait_done(c, job["id"], timeout=240)
    assert done["status"] == "done", done
    return fs.get_result(done["result_key"])["payload"], done["result_key"]


def test_payload_contains_real_values_for_everything_the_ui_displays(real):
    c, hdr, svc = real
    p, key = run_job(c, svc)
    t = p["training"]
    assert isinstance(t["train_rows"], int) and t["train_rows"] > 100 and isinstance(t["test_rows"], int) and t["test_rows"] > 30
    assert isinstance(t["lookback"], int) and 0 < t["train_test_split"] < 1 and isinstance(t["seed"], int)
    assert t["lstm_source"] == "trained this run" and t["epochs"] == 1
    assert t["lstm_training_data_ends"] < t["first_evaluated_date"]                    # provenance: train strictly before eval
    assert p["final_price"] > 0 and set(p["components"]) >= {"linear_regression", "lstm"}
    assert p["weights_used"].get("llm_panel", 0) == 0                                  # no AI component in the number
    ie = p["evaluation"]["interval_evaluation"]
    assert ie["status"] == "ok" and ie["n_evaluated"] >= 30 and p["interval"]["status"] == "ok"
    assert p["interval"]["calibration"]["end"] <= p["evaluation"]["period"]["end"]       # only past residuals calibrate the forecast
    assert p["interval"]["low"] < p["final_price"] < p["interval"]["high"]
    assert p["evaluation"]["status"] == "ok" and p["evaluation"]["out_of_sample"] is True


def test_second_run_reuses_the_checkpoint_with_identical_provenance_and_no_retraining(real, monkeypatch):
    c, hdr, svc = real
    from models.lstm_model import LSTMPredictor
    n = {"t": 0}
    orig = LSTMPredictor.train
    monkeypatch.setattr(LSTMPredictor, "train", lambda self, *a, **k: (n.__setitem__("t", n["t"] + 1), orig(self, *a, **k))[1])
    p1, _ = run_job(c, svc)
    fs._RESULTS.clear()                                                                  # force a recompute, keep the checkpoint
    p2, _ = run_job(c, svc)
    assert n["t"] == 1 and p1["training"]["lstm_source"] == "trained this run" and p2["training"]["lstm_source"] == "reused checkpoint"
    assert p1["training"]["lstm_training_data_ends"] == p2["training"]["lstm_training_data_ends"]
    assert p2["training"]["epochs"] is None


def test_backtest_runs_on_the_same_out_of_sample_frame_and_respects_the_window(real):
    c, hdr, svc = real
    p, key = run_job(c, svc)
    r = fs.run_backtest_job(key, {"model": "lr_lstm_ensemble", "buy_threshold_pct": 0.05, "sell_threshold_pct": -0.05,
                                  "execution": "next_open", "slippage_bps": 2})
    start = p["evaluation"]["period"]["start"]
    assert r["window"]["start"] >= start and r["execution"] == "next_open" and "OPEN" in r["assumptions"]
    assert len(r["benchmark_curve"]) == len(r["equity_curve"]) == len(r["drawdown_curve"])
    with pytest.raises(ValueError):
        fs.run_backtest_job(key, {"model": "lr_lstm_ensemble", "start": "2099-01-01"})


def test_forecast_never_uses_a_checkpoint_trained_on_its_own_evaluation_window(real, monkeypatch):
    """A historical cutoff (less data) must retrain: the earlier checkpoint saw days that are now 'future'."""
    c, hdr, svc = real
    import data_sources.yfinance_source as ys
    run_job(c, svc)
    monkeypatch.setattr(ys, "fetch", lambda *a, **k: fixture_bars(400, seed=3, start="2024-01-02").iloc[:300])
    svc.history_cache._d.clear(); fs._RESULTS.clear()
    p, _ = run_job(c, svc)
    assert p["training"]["lstm_source"] == "trained this run"
    assert p["training"]["lstm_training_data_ends"] < p["training"]["first_evaluated_date"]
