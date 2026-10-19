import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "webapp"), ROOT]

import perf
import forecast_service as fs
from tests.test_v17_webapp import env                                  # noqa: F401
from tests.test_v18_forecast_e2e import real, run_job                  # noqa: F401


def test_perf_only_reports_measured_values_and_is_bounded():
    perf.reset()
    assert perf.snapshot() == {}
    for i in range(700):
        perf.record("x", float(i))
    s = perf.snapshot()["x"]
    assert s["samples"] == 700 and s["window"] == 500 and s["max_ms"] == 699.0 and s["median_ms"] > 0 and s["p95_ms"] >= s["median_ms"]
    with perf.timed("slept"):
        time.sleep(0.05)
    assert 40 <= perf.snapshot()["slept"]["last_ms"] < 500


def test_health_shows_no_performance_numbers_before_anything_ran_then_real_ones(env):
    c, hdr, provider, svc = env
    perf.reset()
    assert c.get("/api/v2/health").get_json()["performance"].keys() <= {"http.api_v2.health"}      # only the health call itself
    c.get("/api/v2/quote?symbol=AAPL&force=1"); c.get("/api/v2/history?symbol=AAPL&period=1y")
    p = c.get("/api/v2/health").get_json()["performance"]
    for key in ("quote.upstream.yahoo", "history.upstream", "http.api_v2.quote", "http.api_v2.history"):
        assert p[key]["samples"] >= 1 and p[key]["median_ms"] >= 0, key


def test_pipeline_and_job_stages_are_measured_for_real_runs(real):
    c, hdr, svc = real
    perf.reset()
    run_job(c, svc)                                                   # real LR + LSTM training on the fixture
    fs._RESULTS.clear()
    run_job(c, svc)                                                   # checkpoint reuse
    p = perf.snapshot()
    for key in ("pipeline.feature_engineering_lstm", "pipeline.lr_train_eval", "pipeline.lstm_train", "pipeline.lstm_checkpoint_load",
                "pipeline.lstm_inference", "forecast.job_total"):
        assert key in p and p[key]["samples"] >= 1, key
    assert p["pipeline.lstm_train"]["samples"] == 1                   # trained once, then reused
    assert p["pipeline.lstm_checkpoint_load"]["median_ms"] < p["pipeline.lstm_train"]["median_ms"]
    key = fs.get_job  # noqa
    r = fs.run_backtest_job(next(iter(fs._RESULTS)), {"model": "lr_lstm_ensemble", "buy_threshold_pct": 0.05, "sell_threshold_pct": -0.05})
    assert perf.snapshot()["backtest.run"]["samples"] == 1 and r["execution"] == "next_open"


# ---- background warm-up
def test_warm_up_runs_records_its_own_measured_time_and_leaves_no_state_behind(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path / "ck"))
    perf.reset(); before = len(fs._RESULTS)
    r = fs.warm_up()
    assert r["ok"] is True and r["ms"] > 0 and perf.snapshot()["warmup.total"]["samples"] == 1
    assert len(fs._RESULTS) == before and not (tmp_path / "ck").exists()          # throw-away model: nothing saved, nothing cached


def test_a_failing_warm_up_is_harmless(monkeypatch):
    from models import lstm_model
    monkeypatch.setattr(lstm_model.LSTMPredictor, "train", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom key=AIzaSyWARM0000000000000000000000000")))
    r = fs.warm_up()
    assert r["ok"] is False                                                        # returns normally; the first forecast just pays the cost


def test_warm_up_is_started_only_by_the_server_entry_point_never_by_importing_the_app():
    src = open(os.path.join(ROOT, "webapp/app.py")).read()
    main = src[src.index('if __name__ == "__main__":'):]
    assert "forecast_service.warm_up" in main and "daemon=True" in main
    assert "forecast_service.warm_up" not in src[:src.index('if __name__ == "__main__":')]
    assert "WARMUP_ON_START" in open(os.path.join(ROOT, ".env.example")).read()
