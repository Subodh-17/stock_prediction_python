import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "webapp"))
sys.path.insert(0, ROOT)

import config
import market_data as md
import forecast_service as fs
from tests.v17_helpers import fixture_bars
from tests.test_v17_market_data import Fake, raw

import app as webapp                      # noqa: E402
from webapp import api_v2                 # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PAPER_PORTFOLIO_PATH", str(tmp_path / "paper.json"))
    monkeypatch.setattr(config, "TRADE_LOG_PATH", str(tmp_path / "log.csv"))
    monkeypatch.setattr(config, "PAPER_STARTING_CASH", 100000.0)
    monkeypatch.setattr(config, "WEBAPP_EXPENSIVE_RPM", 1000)
    api_v2._hits.clear()
    provider = Fake("yahoo", raw=raw(price=50.0))
    svc = md.MarketDataService([provider])
    monkeypatch.setattr(md, "_service", svc)
    import data_sources.yfinance_source as ys
    monkeypatch.setattr(ys, "fetch", lambda *a, **k: fixture_bars(200))
    fs.JOBS.clear(); fs._RESULTS.clear(); fs._ACTIVE_BY_KEY.clear()
    c = webapp.app.test_client()
    tok = c.get("/api/v2/meta").get_json()["csrf_token"]
    return c, {"X-CSRF-Token": tok}, provider, svc


# ---- pages and assets
def test_every_page_and_asset_renders(env):
    c, *_ = env
    r = c.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "Run Prediction" in html and 'name="ticker"' in html and 'name="csrf-token"' in html
    for href in re.findall(r'href="#/(\w+)"', html):
        assert href in {"overview", "research", "forecast", "evaluation", "backtest", "ai", "portfolio", "risk", "health", "settings"}
    assert len(set(re.findall(r'href="#/(\w+)"', html))) == 10
    js = open(os.path.join(ROOT, "webapp/static/ws.js"), encoding="utf-8").read()
    for view in set(re.findall(r'href="#/(\w+)"', html)):
        assert f"VIEWS.{view} =" in js, f"nav link #{view} has no implementation"
    assert c.get("/static/ws.js").status_code == 200 and c.get("/static/ws.css").status_code == 200


def test_security_headers_present(env):
    c, *_ = env
    h = c.get("/").headers
    assert "script-src 'self'" in h.get("Content-Security-Policy", "") and h.get("X-Content-Type-Options") == "nosniff"


def test_blank_secret_key_from_copied_env_example_no_longer_crashes():
    code = ("import os,sys; os.environ['WEBAPP_SECRET_KEY']=''; sys.path[:0]=['webapp','.']\n"
            "import app; r=app.app.test_client().get('/'); print(r.status_code, bool(app.app.secret_key))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120,
                         env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})
    assert out.stdout.strip().endswith("200 True"), out.stderr[-500:]


def test_javascript_has_no_syntax_errors():
    import shutil
    if not shutil.which("node"):
        pytest.skip("node not installed")
    r = subprocess.run(["node", "--check", os.path.join(ROOT, "webapp/static/ws.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_static_assets_contain_no_secrets_and_no_demo_content(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "AIzaSyTESTSECRETVALUE1234567890abcdef")
    blob = ""
    for n in ("ws.js", "ws.css"):
        blob += open(os.path.join(ROOT, "webapp/static", n), encoding="utf-8").read()
    blob += open(os.path.join(ROOT, "webapp/templates/workstation.html"), encoding="utf-8").read()
    assert config.GOOGLE_API_KEY not in blob
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,}", blob)
    low = blob.lower().replace("never substitutes sample data", "")   # honesty statement, not content
    for bad in ("demo data", "sample data", "mock data", "lorem", "fake "):
        assert bad not in low, bad
    # no hardcoded market numbers: no price-like literals assigned to tickers
    assert not re.search(r"\b(AAPL|MSFT|TSLA|RELIANCE)\b[^'\"\n]{0,20}\d{2,}\.\d{2}", blob)


def test_live_badge_is_only_produced_by_the_freshness_map():
    js = open(os.path.join(ROOT, "webapp/static/ws.js"), encoding="utf-8").read()
    occurrences = [m.start() for m in re.finditer(r"'Live'", js)]
    assert len(occurrences) == 1                                   # the single label in FRESH, driven by q.freshness
    assert "freshness || 'unavailable'" in js


def test_no_innerhtml_in_ui_code():
    js = open(os.path.join(ROOT, "webapp/static/ws.js"), encoding="utf-8").read()
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)          # comments may name the rule; code may not break it
    assert "innerHTML" not in js and "document.write" not in js and "eval(" not in js


# ---- CSRF, validation, rate limiting
def test_state_changing_routes_require_csrf_token(env):
    c, hdr, *_ = env
    for path in ("/api/v2/forecast", "/api/v2/backtest", "/api/v2/ai/test", "/api/v2/ai/research", "/api/v2/paper/order", "/api/v2/paper/reset", "/api/v2/jobs/x/cancel"):
        assert c.post(path, json={}).status_code == 403, path
        assert c.post(path, json={}, headers={"X-CSRF-Token": "wrong"}).status_code == 403, path
    r = c.post("/api/v2/paper/order", json={}, headers={**hdr, "Origin": "http://evil.example"})
    assert r.status_code == 403


def test_no_route_can_place_a_live_order(env):
    rules = [r.rule for r in webapp.app.url_map.iter_rules() if r.rule.startswith("/api/v2")]
    assert not any("live" in r for r in rules)
    c, hdr, *_ = env
    assert c.get("/api/v2/meta").get_json()["trading"]["live_orders_via_this_api"] is False


def test_invalid_symbol_and_params_are_400_not_500(env):
    c, hdr, *_ = env
    assert c.get("/api/v2/quote?symbol=../../etc").status_code == 400
    assert c.get("/api/v2/quote").status_code == 400
    assert c.get("/api/v2/quotes").status_code == 400
    assert c.get("/api/v2/history?symbol=AAPL&period=1y;drop").status_code == 400
    assert c.post("/api/v2/forecast", json={"symbol": "AAPL", "epochs": 99999}, headers=hdr).status_code == 400


def test_expensive_routes_are_rate_limited_with_retry_after(env, monkeypatch):
    c, hdr, *_ = env
    monkeypatch.setattr(config, "WEBAPP_EXPENSIVE_RPM", 2)
    codes = [c.post("/api/v2/ai/test", json={"provider": "gemini"}, headers=hdr) for _ in range(4)]
    assert [r.status_code for r in codes][-1] == 429 and codes[-1].headers.get("Retry-After")


def test_errors_do_not_leak_paths_or_stack_traces(env, monkeypatch):
    c, hdr, provider, svc = env
    monkeypatch.setattr(svc, "get_quote", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("/home/secret/path.py boom")))
    r = c.get("/api/v2/quote?symbol=AAPL")
    body = r.get_data(as_text=True)
    assert r.status_code == 500 and "/home/secret" not in body and "Traceback" not in body


# ---- quotes and honesty
def test_quote_endpoint_reports_freshness_and_never_live_for_delayed_source(env):
    c, hdr, provider, svc = env
    q = c.get("/api/v2/quote?symbol=AAPL&force=1").get_json()
    assert q["freshness"] in ("delayed", "stale") and q["provider"] == "yahoo" and q["realtime_eligible"] is False
    assert q["source_ts"] and q["retrieved_ts"]


def test_quote_outage_returns_unavailable_without_numbers(env):
    c, hdr, provider, svc = env
    provider._exc = md.ProviderError("upstream down")
    q = c.get("/api/v2/quote?symbol=NEWCO&force=1").get_json()
    assert q["freshness"] == "unavailable" and q["price"] is None and "upstream down" in q["error"]


def test_news_without_feeds_is_an_honest_empty_state(env, monkeypatch):
    c, *_ = env
    monkeypatch.setattr(config, "NEWS_FEEDS", "")
    n = c.get("/api/v2/news?symbol=AAPL").get_json()
    assert n["status"] == "not_configured" and n["items"] == []


def test_history_returns_real_bars_with_indicators_and_provenance(env):
    c, *_ = env
    h = c.get("/api/v2/history?symbol=NSE:TCS&period=1y").get_json()
    assert h["provider"] == "yahoo" and h["adjusted"] and len(h["bars"]) == 200 and h["instrument"]["exchange"] == "NSE"
    assert len(h["indicators"]["series"]["sma20"]) == 200 and h["indicators"]["series"]["sma20"][0] is None


# ---- paper trading
def order(c, hdr, **kw):
    body = {"symbol": "AAPL", "side": "BUY", "quantity": 10, "confirm": True, **kw}
    return c.post("/api/v2/paper/order", json=body, headers=hdr)


def test_paper_order_requires_explicit_confirmation(env):
    c, hdr, *_ = env
    assert order(c, hdr, confirm=False).status_code == 400
    assert c.get("/api/v2/portfolio").get_json()["positions"] == []


def test_paper_order_uses_fresh_quote_and_is_labelled_simulated(env):
    c, hdr, provider, svc = env
    r = order(c, hdr)
    j = r.get_json()
    assert r.status_code == 200 and j["simulated"] is True and j["price_used"] == 50.0 and j["quote_provider"] == "yahoo"
    p = c.get("/api/v2/portfolio").get_json()
    assert "PAPER" in p["kind"] and p["positions"][0]["quantity"] == 10 and p["cash"] == pytest.approx(100000 - 500)
    assert p["live_account"]["connected_via_this_ui"] is False


def test_paper_order_refused_without_a_usable_quote(env):
    c, hdr, provider, svc = env
    provider._exc = md.ProviderError("down")
    r = order(c, hdr, symbol="ZZZZ")
    assert r.status_code == 409 and r.get_json()["error"] == "no_fresh_quote"
    stale = Fake("yahoo", raw=raw(ts=datetime.now(timezone.utc) - timedelta(days=30), dtype="delayed"))
    svc.providers[:] = [stale]
    assert order(c, hdr, symbol="OLDQ").status_code == 409
    assert c.get("/api/v2/portfolio").get_json()["positions"] == []


def test_risk_limits_still_block_oversized_paper_orders(env):
    c, hdr, *_ = env
    r = order(c, hdr, quantity=10_000_000)
    assert r.status_code == 422 and r.get_json()["error"] == "risk_blocked"


def test_paper_reset_sets_user_balance_and_needs_confirm(env):
    c, hdr, *_ = env
    assert c.post("/api/v2/paper/reset", json={"starting_cash": 5000}, headers=hdr).status_code == 400
    assert c.post("/api/v2/paper/reset", json={"starting_cash": 5000, "confirm": True}, headers=hdr).status_code == 200
    assert c.get("/api/v2/portfolio").get_json()["cash"] == 5000


# ---- forecast jobs: background, dedupe, no retrain on refresh
def _fake_run(counter, gate=None):
    def run(job, raw_symbol, inst, period, epochs, svc, walk_forward=False):
        counter["n"] += 1
        if gate:
            gate.wait(5)
        return {"instrument": inst.to_dict(), "final_price": 1.0}, {"frame": None, "df": None}
    return run


def wait_done(c, jid, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/v2/jobs/{jid}").get_json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_forecast_runs_in_background_and_identical_requests_share_one_job(env, monkeypatch):
    import threading
    c, hdr, *_ = env
    n, gate = {"n": 0}, threading.Event()
    monkeypatch.setattr(fs, "_run_pipeline_and_evaluate", _fake_run(n, gate))
    a = c.post("/api/v2/forecast", json={"symbol": "AAPL", "period": "2y", "epochs": 5}, headers=hdr)
    b = c.post("/api/v2/forecast", json={"symbol": "AAPL", "period": "2y", "epochs": 5}, headers=hdr)
    assert a.status_code == 202 and a.get_json()["id"] == b.get_json()["id"]
    assert c.get("/api/v2/health").status_code == 200           # server stays responsive while a job runs
    gate.set()
    assert wait_done(c, a.get_json()["id"])["status"] == "done" and n["n"] == 1


def test_quote_refresh_and_repeat_requests_never_retrain(env, monkeypatch):
    c, hdr, provider, svc = env
    n = {"n": 0}
    monkeypatch.setattr(fs, "_run_pipeline_and_evaluate", _fake_run(n))
    j1 = c.post("/api/v2/forecast", json={"symbol": "AAPL", "period": "2y", "epochs": 5}, headers=hdr).get_json()
    wait_done(c, j1["id"])
    for _ in range(5):
        c.get("/api/v2/quote?symbol=AAPL&force=1")             # manual refreshes
    j2 = c.post("/api/v2/forecast", json={"symbol": "AAPL", "period": "2y", "epochs": 5}, headers=hdr).get_json()
    assert j2["status"] == "done" and j2["timings"].get("cached") is True and n["n"] == 1


def test_failed_job_reports_a_bounded_redacted_error(env, monkeypatch):
    c, hdr, *_ = env
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "AIzaSyLEAKME_1234567890_abcdefghijk")
    def boom(*a, **k):
        raise RuntimeError("failed with key AIzaSyLEAKME_1234567890_abcdefghijk " + "x" * 2000)
    monkeypatch.setattr(fs, "_run_pipeline_and_evaluate", boom)
    j = c.post("/api/v2/forecast", json={"symbol": "AAPL", "period": "2y", "epochs": 5}, headers=hdr).get_json()
    done = wait_done(c, j["id"])
    assert done["status"] == "failed" and "AIzaSyLEAKME" not in done["error"] and len(done["error"]) <= 400


def test_cancelling_a_queued_job_prevents_it_from_running(env, monkeypatch):
    import threading
    c, hdr, *_ = env
    n, gate = {"n": 0}, threading.Event()
    monkeypatch.setattr(fs, "_run_pipeline_and_evaluate", _fake_run(n, gate))
    first = c.post("/api/v2/forecast", json={"symbol": "AAPL", "period": "2y", "epochs": 5}, headers=hdr).get_json()
    second = c.post("/api/v2/forecast", json={"symbol": "MSFT", "period": "2y", "epochs": 5}, headers=hdr).get_json()
    assert c.post(f"/api/v2/jobs/{second['id']}/cancel", json={}, headers=hdr).get_json()["cancel_requested"] is True
    gate.set(); wait_done(c, first["id"]); time.sleep(0.3)
    assert c.get(f"/api/v2/jobs/{second['id']}").get_json()["status"] == "cancelled" and n["n"] == 1


# ---- AI endpoints and health
def test_ai_status_never_returns_key_values(env, monkeypatch):
    c, *_ = env
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "AIzaSyVERYSECRET_0123456789abcdefghij")
    body = c.get("/api/v2/ai/status").get_data(as_text=True) + c.get("/api/v2/settings").get_data(as_text=True) + c.get("/api/v2/health").get_data(as_text=True)
    assert "VERYSECRET" not in body


def test_ai_research_requires_a_computed_forecast(env):
    c, hdr, *_ = env
    r = c.post("/api/v2/ai/research", json={"key": "nope"}, headers=hdr)
    assert r.status_code == 404


def test_health_reports_measured_values_only(env):
    c, hdr, provider, svc = env
    h0 = c.get("/api/v2/health").get_json()["market_data"]["measured_latency_ms"]
    assert h0["quote"]["samples"] == 0 and h0["quote"]["median"] is None          # nothing measured -> nothing shown
    c.get("/api/v2/quote?symbol=AAPL&force=1")
    h1 = c.get("/api/v2/health").get_json()["market_data"]["measured_latency_ms"]
    assert h1["quote"]["samples"] == 1 and h1["quote"]["median"] is not None


def test_legacy_routes_still_work(env):
    c, *_ = env
    assert c.get("/results").status_code in (200, 302)
    assert c.get("/health").status_code == 200 if any(r.rule == "/health" for r in webapp.app.url_map.iter_rules()) else True


def test_legacy_expensive_routes_are_now_rate_limited(env, monkeypatch):
    c, *_ = env
    monkeypatch.setattr(config, "WEBAPP_EXPENSIVE_RPM", 2)
    codes = [c.post("/api/predict", json={}).status_code for _ in range(4)]
    assert codes[-1] == 429 and 429 not in codes[:2]
