import io
import logging
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "webapp"), ROOT]

import config
import app as webapp
from tests.test_v17_webapp import env                       # noqa: F401

KEY = "MOCK_GOOGLE_API_KEY_FIXTURE_9876543210"


def _post_rules():
    return sorted((r.rule, r.endpoint) for r in webapp.app.url_map.iter_rules() if "POST" in r.methods and not r.rule.startswith("/static"))


def test_every_post_route_refuses_a_request_without_csrf_proof(env):
    """Route inventory: no state-changing route may succeed for an anonymous, token-less request. The only documented
    exemption is the legacy read-only JSON compute API (/api/predict|signal|backtest), asserted separately below."""
    c, *_ = env
    exempt = {"/api/predict", "/api/signal", "/api/backtest"}
    results = {}
    for rule, ep in _post_rules():
        if rule in exempt or "<" in rule:
            path = rule.replace("<job_id>", "x")
            if rule in exempt:
                continue
        else:
            path = rule
        r = webapp.app.test_client().post(path, json={}, data=None) if path.startswith("/api/v2") else webapp.app.test_client().post(path, data={})
        results[path] = r.status_code
        assert r.status_code in (400, 401, 403, 404, 405, 413, 422, 429), (path, r.status_code)
    assert results, "no POST routes found"


def test_legacy_json_api_cannot_reach_trading_code():
    import inspect
    for fn in (webapp.api_predict, webapp.api_signal, webapp.api_backtest, webapp._compute_api_predict):
        src = inspect.getsource(fn)
        assert not re.search(r"place_order|execute_trade|get_broker|trade_executor|UpstoxBroker|GrowwBroker", src), fn.__name__


def test_live_trade_is_refused_at_every_gate_and_never_instantiates_a_broker(env, monkeypatch):
    """With a forecast result in the session (so the request really reaches the live gates): not enabled -> 403;
    enabled but no operator token configured -> 403; enabled + token but wrong operator secret -> 403 and
    no broker object is ever created."""
    import time
    c, hdr, *_ = env
    from trading.brokers import upstox_broker, groww_broker
    created = []
    monkeypatch.setattr(upstox_broker.UpstoxBroker, "__init__", lambda self, *a, **k: created.append("upstox"))
    monkeypatch.setattr(groww_broker.GrowwBroker, "__init__", lambda self, *a, **k: created.append("groww"))
    rid = webapp._store_result({"created_at": time.time(), "ticker": "AAPL", "final_ensemble": {}})
    with c.session_transaction() as s:
        s["result_id"] = rid
    tok = hdr["X-CSRF-Token"]
    form = lambda b: {"csrf_token": tok, "mode": "live", "broker": b, "operator_secret": "wrong-secret"}

    monkeypatch.setattr(config, "ENABLE_LIVE_TRADING", False)
    for b in ("upstox", "groww"):
        assert c.post("/trade", data=form(b)).status_code == 403
    monkeypatch.setattr(config, "ENABLE_LIVE_TRADING", True)
    monkeypatch.setattr(config, "WEBAPP_LIVE_TRADING_TOKEN", "")
    for b in ("upstox", "groww"):
        assert c.post("/trade", data=form(b)).status_code == 403
    monkeypatch.setattr(config, "WEBAPP_LIVE_TRADING_TOKEN", "a-long-operator-secret-of-at-least-32-chars!!")
    for b in ("upstox", "groww"):
        r = c.post("/trade", data=form(b))
        assert r.status_code in (400, 401, 403), r.status_code
    assert created == []                                                                    # nothing was ever constructed
    assert c.post("/trade", data={**form("paper"), "mode": "paper", "csrf_token": "bad"}).status_code in (400, 403)


def test_cookie_size_bind_and_cors_settings():
    assert webapp.app.config["SESSION_COOKIE_HTTPONLY"] is True and webapp.app.config["SESSION_COOKIE_SAMESITE"] in ("Lax", "Strict")
    assert 0 < webapp.app.config["MAX_CONTENT_LENGTH"] <= 1024 * 1024
    assert 'host="127.0.0.1"' in open(os.path.join(ROOT, "webapp/app.py")).read()           # local by default
    c = webapp.app.test_client()
    r = c.get("/api/v2/meta", headers={"Origin": "http://evil.example"})
    assert "Access-Control-Allow-Origin" not in r.headers                                  # no CORS: other sites cannot read the token
    assert "HttpOnly" in c.get("/").headers.get("Set-Cookie", "") or "session" not in c.get("/").headers.get("Set-Cookie", "")


def test_oversized_requests_are_rejected(env):
    c, hdr, *_ = env
    r = c.post("/api/v2/ai/test", data="x" * (webapp.app.config["MAX_CONTENT_LENGTH"] + 10), headers={**hdr, "Content-Type": "application/json"})
    assert r.status_code == 413


def test_api_v2_responses_are_never_cacheable_and_carry_security_headers(env):
    c, *_ = env
    for path in ("/api/v2/meta", "/api/v2/portfolio", "/api/v2/ai/status"):
        r = c.get(path)
        assert r.headers.get("Cache-Control") == "no-store" and r.headers.get("X-Content-Type-Options") == "nosniff"


def test_project_loggers_redact_secrets_inside_tracebacks():
    import forecast_service, logging_utils
    from webapp import api_v2
    config.GOOGLE_API_KEY = KEY
    try:
        for lg in (forecast_service._log, api_v2._log):
            assert lg.name.startswith("stockpred."), lg.name
            stream = io.StringIO(); h = logging.StreamHandler(stream)
            h.setFormatter(logging.Formatter("%(message)s"))
            lg.addHandler(h)
            try:
                try:
                    raise RuntimeError(f"upstream said key={KEY} bad")
                except RuntimeError:
                    lg.error("failure: %s", f"RuntimeError key={KEY}", exc_info=True)
            finally:
                lg.removeHandler(h)
            out = stream.getvalue()
            assert KEY not in out and "failure" in out
    finally:
        config.GOOGLE_API_KEY = ""


def test_a_failing_job_never_leaks_the_key_into_status_or_logs(env, monkeypatch):
    import forecast_service as fs
    from tests.test_v17_webapp import wait_done
    c, hdr, *_ = env
    monkeypatch.setattr(config, "GOOGLE_API_KEY", KEY)
    def boom(*a, **k):
        raise RuntimeError(f"Authorization failed for key {KEY}")
    monkeypatch.setattr(fs, "_run_pipeline_and_evaluate", boom)
    j = c.post("/api/v2/forecast", json={"symbol": "AAPL", "period": "2y", "epochs": 5}, headers=hdr).get_json()
    done = wait_done(c, j["id"])
    assert done["status"] == "failed" and KEY not in str(done) and "Authorization failed" not in str(done)


def test_live_trading_is_off_by_default_and_paper_state_is_a_separate_file():
    """Checks the SHIPPED default in a clean subprocess: other tests may mutate config in-process."""
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ENABLE_LIVE", "WEBAPP_LIVE"))}
    out = subprocess.run([sys.executable, "-c", "import config; print(config.ENABLE_LIVE_TRADING, config.PAPER_PORTFOLIO_PATH)"],
                         cwd=ROOT, capture_output=True, text=True, env=env, timeout=120).stdout.split()
    assert out[0] == "False" and "paper" in os.path.basename(out[1]).lower()
    assert "live" not in os.path.basename(out[1]).lower()
