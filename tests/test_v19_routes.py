import os
import re
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "webapp"), ROOT]

import config
import forecast_service as fs
import app as webapp
from tests.test_v17_webapp import env, wait_done                 # noqa: F401
from tests.test_v18_jobs import client                           # noqa: F401  (fixture: env + short legacy wait)


def slow(kind_payload, gate):
    def f(*a):
        gate.wait(10)
        a[-1]["_payload"] = kind_payload
        return "x"
    return f


@pytest.mark.parametrize("path,fn,body,payload", [
    ("/api/signal", "_compute_api_signal", {"ticker": "AAPL"}, {"signal": {"action": "HOLD"}}),
    ("/api/backtest", "_compute_api_backtest", {"ticker": "AAPL"}, {"total_return_pct": 1.0}),
])
def test_signal_and_backtest_run_as_background_jobs_and_async_returns_immediately(client, monkeypatch, path, fn, body, payload):
    c, hdr = client
    gate = threading.Event()
    monkeypatch.setattr(webapp, fn, slow(payload, gate))
    t0 = time.time()
    r = c.post(path, json={**body, "async": True})
    assert r.status_code == 202 and time.time() - t0 < 5
    j = r.get_json()
    assert re.fullmatch(r"[0-9a-f]{24}", j["job_id"]) and j["result_url"] == f"/api/jobs/{j['job_id']}/result"
    assert c.get(j["result_url"]).status_code == 202                       # not finished: no fake completion
    gate.set(); wait_done(c, j["job_id"])
    done = c.get(j["result_url"])
    assert done.status_code == 200 and done.get_json() == payload
    assert c.get(f"/api/jobs/{j['job_id']}/result").status_code == 200 and c.get("/api/jobs/zzzz/result").status_code == 404


def test_without_async_the_old_contract_holds_within_the_bounded_wait(client, monkeypatch):
    c, hdr = client
    monkeypatch.setattr(config, "LEGACY_PREDICT_WAIT_SECONDS", 30)
    monkeypatch.setattr(webapp, "_compute_api_signal", slow({"signal": "ok"}, threading.Event()))
    gate = threading.Event(); gate.set()
    monkeypatch.setattr(webapp, "_compute_api_signal", slow({"signal": "ok"}, gate))
    r = c.post("/api/signal", json={"ticker": "AAPL"})
    assert r.status_code == 200 and r.get_json() == {"signal": "ok"}


def test_background_failures_keep_their_original_status_and_error_code(client, monkeypatch):
    c, hdr = client
    def boom(*a):
        raise fs.TaskError("Not enough rows.", 422, "insufficient_data")
    monkeypatch.setattr(webapp, "_compute_api_backtest", boom)
    r = c.post("/api/backtest", json={"ticker": "AAPL"})
    assert r.status_code == 422 and r.get_json() == {"error": "insufficient_data", "message": "Not enough rows."}
    monkeypatch.setattr(webapp, "_compute_api_signal", lambda *a: (_ for _ in ()).throw(RuntimeError("secret path /srv/x.py")))
    r = c.post("/api/signal", json={"ticker": "AAPL"})
    assert r.status_code == 500 and "/srv/x.py" not in r.get_data(as_text=True)


def test_identical_signal_requests_share_one_job(client, monkeypatch):
    c, hdr = client
    gate, calls = threading.Event(), []
    def comp(*a):
        calls.append(1); gate.wait(10); a[-1]["_payload"] = {"s": 1}; return "x"
    monkeypatch.setattr(webapp, "_compute_api_signal", comp)
    ids = {c.post("/api/signal", json={"ticker": "AAPL", "async": True}).get_json()["job_id"] for _ in range(5)}
    gate.set(); time.sleep(0.4)
    assert len(ids) == 1 and len(calls) == 1


def test_a_slow_job_never_blocks_quotes_health_or_other_endpoints(client, monkeypatch):
    """Regression for 'synchronous route blocking': while a long job runs, the rest of the app stays responsive."""
    c, hdr = client
    gate = threading.Event()
    monkeypatch.setattr(webapp, "_compute_api_signal", slow({"s": 1}, gate))
    monkeypatch.setattr(webapp, "_compute_legacy_predict", lambda p: (gate.wait(10), webapp._store_result({"created_at": time.time()}))[1])
    assert c.post("/api/signal", json={"ticker": "AAPL", "async": True}).status_code == 202
    assert c.post("/predict", data={"ticker": "AAPL", "async": "1", "csrf_token": hdr["X-CSRF-Token"]}).status_code == 202
    for path in ("/api/v2/quote?symbol=AAPL&force=1", "/api/v2/health", "/api/v2/history?symbol=AAPL&period=1y", "/api/v2/ai/status", "/api/health"):
        t0 = time.time()
        assert c.get(path).status_code == 200 and time.time() - t0 < 3, path
    gate.set()


def test_slow_ai_research_does_not_block_quotes(env, monkeypatch):
    c, hdr, provider, svc = env
    import research
    from tests.test_v17_webapp import fs as _fs
    gate = threading.Event()
    monkeypatch.setattr(research, "generate", lambda *a, **k: (gate.wait(3), {"status": "unavailable", "reason": "x", "attempts": []})[1])
    _fs._RESULTS["k"] = {"payload": {"instrument": {"yahoo_symbol": "AAPL", "symbol": "AAPL", "exchange": "US/UNSPECIFIED"},
                         "components": {}, "final_price": 1.0, "pct_change": 0.0, "as_of_bar": "2025-01-01",
                         "target": {"lstm": "t"}, "interval": {"status": "x"}, "evaluation": {"models": [], "period": {}, "n_obs": 0}}, "at": time.time()}
    out = {}
    th = threading.Thread(target=lambda: out.update(r=c.post("/api/v2/ai/research", json={"key": "k"}, headers=hdr)))
    th.start(); time.sleep(0.3)
    t0 = time.time()
    assert c.get("/api/v2/quote?symbol=AAPL").status_code == 200 and time.time() - t0 < 1.5
    th.join(); gate.set()
    assert out["r"].status_code == 200
