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
from tests.test_v17_webapp import env, wait_done                   # noqa: F401  (fixtures reused)
from tests.v17_helpers import fixture_bars


@pytest.fixture
def client(env, monkeypatch):
    c, hdr, provider, svc = env
    monkeypatch.setattr(config, "LEGACY_PREDICT_WAIT_SECONDS", 0.15)
    return c, hdr


FORM = {"ticker": "AAPL", "period": "1y", "epochs": "1"}


def post_predict(c, hdr, **kw):
    return c.post("/predict", data={**FORM, **kw, "csrf_token": hdr["X-CSRF-Token"]})


def job_id(resp):
    return re.search(r'data-job-id="(\w+)"', resp.get_data(as_text=True)).group(1)


def fake_compute(gate=None, calls=None, fail=None):
    def f(p):
        if calls is not None:
            calls.append(p["ticker"])
        if gate:
            gate.wait(10)
        if fail:
            raise fail
        return webapp._store_result({"created_at": time.time(), "ticker": p["ticker"]})
    return f


def test_slow_predict_returns_202_with_a_job_id_promptly_then_claim_redirects(client, monkeypatch):
    c, hdr = client
    gate = threading.Event()
    monkeypatch.setattr(webapp, "_compute_legacy_predict", fake_compute(gate))
    t0 = time.time()
    r = post_predict(c, hdr)
    assert r.status_code == 202 and time.time() - t0 < 2 and 'id="job"' in r.get_data(as_text=True)
    jid = job_id(r)
    st = c.get(f"/api/v2/jobs/{jid}").get_json()
    assert st["status"] in ("queued", "running") and st["kind"] == "legacy_predict"
    assert c.get(f"/predict/claim/{jid}").status_code == 409            # not finished -> not claimable, no fake completion
    gate.set()
    assert wait_done(c, jid)["status"] == "done"
    claim = c.get(f"/predict/claim/{jid}")
    assert claim.status_code == 302 and claim.headers["Location"].endswith("/results")
    with c.session_transaction() as s:
        assert s["result_id"] == fs.get_job(jid)["result_key"]


def test_fast_completion_keeps_the_classic_immediate_redirect(client, monkeypatch):
    c, hdr = client
    monkeypatch.setattr(webapp, "_compute_legacy_predict", fake_compute())
    r = post_predict(c, hdr)
    assert r.status_code == 302 and r.headers["Location"].endswith("/results")


def test_failed_job_shows_its_error_and_never_an_older_forecast(client, monkeypatch):
    c, hdr = client
    monkeypatch.setattr(webapp, "_compute_legacy_predict", fake_compute())
    assert post_predict(c, hdr).status_code == 302                       # a first, successful forecast exists
    with c.session_transaction() as s:
        old = s["result_id"]
    gate = threading.Event()
    monkeypatch.setattr(webapp, "_compute_legacy_predict",
                        fake_compute(gate, fail=fs.TaskError("No data for ZZZ.", 400)))
    r = post_predict(c, hdr, ticker="ZZZ")
    assert r.status_code == 202
    jid = job_id(r); gate.set()
    assert wait_done(c, jid)["status"] == "failed"
    claim = c.get(f"/predict/claim/{jid}")
    assert claim.status_code == 400 and "No data for ZZZ." in claim.get_data(as_text=True)
    with c.session_transaction() as s:
        assert s.get("result_id") != old and "result_id" not in s        # the stale forecast is not left behind
    assert c.get("/results").status_code == 302                           # redirected away, nothing stale rendered


def test_unexpected_errors_are_redacted_and_generic(client, monkeypatch):
    c, hdr = client
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "AIzaSyLEAKME_1234567890_abcdefghijk")
    monkeypatch.setattr(webapp, "_compute_legacy_predict", fake_compute(fail=RuntimeError("boom AIzaSyLEAKME_1234567890_abcdefghijk /srv/app.py")))
    r = post_predict(c, hdr)
    body = r.get_data(as_text=True)
    assert r.status_code == 202 or r.status_code == 500
    jid = job_id(r) if r.status_code == 202 else None
    if jid:
        wait_done(c, jid)
        body = c.get(f"/predict/claim/{jid}").get_data(as_text=True) + str(c.get(f"/api/v2/jobs/{jid}").get_json())
    assert "AIzaSyLEAKME" not in body and "/srv/app.py" not in body and "boom" not in body


def test_identical_concurrent_requests_share_one_job(client, monkeypatch):
    c, hdr = client
    gate, calls = threading.Event(), []
    monkeypatch.setattr(webapp, "_compute_legacy_predict", fake_compute(gate, calls))
    ids = set()
    def hit():
        cc = webapp.app.test_client()
        tok = cc.get("/api/v2/meta").get_json()["csrf_token"]
        ids.add(job_id(cc.post("/predict", data={**FORM, "csrf_token": tok})))
    ts = [threading.Thread(target=hit) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    gate.set(); time.sleep(0.4)
    assert len(ids) == 1 and len(calls) == 1


def test_different_parameters_get_different_jobs(client, monkeypatch):
    c, hdr = client
    gate, calls = threading.Event(), []
    monkeypatch.setattr(webapp, "_compute_legacy_predict", fake_compute(gate, calls))
    a, b = job_id(post_predict(c, hdr, epochs="1")), job_id(post_predict(c, hdr, epochs="2"))
    gate.set(); time.sleep(0.4)
    assert a != b and len(calls) == 2


def test_cancelling_a_queued_job_prevents_it_and_claim_reports_cancelled(client, monkeypatch):
    c, hdr = client
    gate, calls = threading.Event(), []
    monkeypatch.setattr(webapp, "_compute_legacy_predict", fake_compute(gate, calls))
    first = job_id(post_predict(c, hdr, ticker="AAA"))
    second = job_id(post_predict(c, hdr, ticker="BBB"))                  # queued behind the single worker
    r = c.post(f"/api/v2/jobs/{second}/cancel", json={}, headers=hdr)
    assert r.get_json()["cancel_requested"] is True
    gate.set(); wait_done(c, first); time.sleep(0.3)
    assert calls == ["AAA"] and c.get(f"/predict/claim/{second}").status_code == 409


def test_running_job_past_the_limit_is_reported_as_timeout_not_forever_running(client, monkeypatch):
    c, hdr = client
    gate = threading.Event()
    monkeypatch.setattr(webapp, "_compute_legacy_predict", fake_compute(gate))
    jid = job_id(post_predict(c, hdr))
    time.sleep(0.2)
    monkeypatch.setattr(config, "JOB_TIMEOUT_SECONDS", 0.05)
    st = c.get(f"/api/v2/jobs/{jid}").get_json()
    gate.set()
    assert st["status"] == "timeout" and st["timed_out"] is True


def test_polling_script_is_bounded_and_waits_for_the_real_status():
    js = open(os.path.join(ROOT, "webapp/static/job_poll.js")).read()
    assert "limitMs" in js and "Date.now() - t0 > limitMs" in js and "/api/v2/jobs/" in js
    assert js.count("claim") >= 1 and "timeout" in js
    assert "innerHTML" not in js


def test_api_predict_returns_202_with_job_id_then_result(client, monkeypatch):
    c, hdr = client
    gate = threading.Event()
    def compute(*a):
        job = a[-1]; gate.wait(10); job["_payload"] = {"ticker": "AAPL", "final_ensemble": {"ok": True}}; return "api_predict"
    monkeypatch.setattr(webapp, "_compute_api_predict", compute)
    r = c.post("/api/predict", json={"ticker": "AAPL"})
    assert r.status_code == 202 and r.get_json()["job_id"]
    jid = r.get_json()["job_id"]
    assert c.get(f"/api/predict/result/{jid}").status_code == 202
    gate.set(); wait_done(c, jid)
    done = c.get(f"/api/predict/result/{jid}")
    assert done.status_code == 200 and done.get_json()["ticker"] == "AAPL"
    assert c.get("/api/predict/result/nope").status_code == 404


def test_api_predict_failure_keeps_its_status_and_code(client, monkeypatch):
    c, hdr = client
    def compute(*a):
        raise fs.TaskError("Not enough data.", 422, "insufficient_data")
    monkeypatch.setattr(webapp, "_compute_api_predict", compute)
    r = c.post("/api/predict", json={"ticker": "AAPL"})
    assert r.status_code == 422 and r.get_json()["error"] == "insufficient_data"


# ---- forecast-service races
def test_simultaneous_identical_forecast_submits_create_exactly_one_job(env, monkeypatch):
    c, hdr, provider, svc = env
    gate, n = threading.Event(), {"n": 0}
    def run(job, raw, inst, period, epochs, svc_, wf=False):
        n["n"] += 1; gate.wait(10)
        return {"final_price": 1.0, "instrument": inst.to_dict()}, {"frame": None, "df": None}
    monkeypatch.setattr(fs, "_run_pipeline_and_evaluate", run)
    ids, barrier = [], threading.Barrier(8)
    def go():
        barrier.wait()
        ids.append(fs.submit_forecast("AAPL", "2y", 5, svc=svc)["id"])
    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    gate.set(); time.sleep(0.5)
    assert len(set(ids)) == 1 and n["n"] == 1


def test_different_epochs_or_walk_forward_never_share_or_overwrite_cache_entries(env, monkeypatch):
    c, hdr, provider, svc = env
    def run(job, raw, inst, period, epochs, svc_, wf=False):
        return {"final_price": float(epochs) + (0.5 if wf else 0), "instrument": inst.to_dict()}, {"frame": None, "df": None}
    monkeypatch.setattr(fs, "_run_pipeline_and_evaluate", run)
    jobs = [fs.submit_forecast("AAPL", "2y", e, svc=svc, walk_forward=w) for e, w in ((3, False), (4, False), (3, True))]
    for j in jobs:
        wait_done(c, j["id"])
    keys = [fs.get_job(j["id"])["result_key"] for j in jobs]
    assert len(set(keys)) == 3
    assert [fs.get_result(k)["payload"]["final_price"] for k in keys] == [3.0, 4.0, 3.5]


def test_html_forms_opt_in_to_the_immediate_202_even_for_a_fast_job(client, monkeypatch):
    c, hdr = client
    monkeypatch.setattr(config, "LEGACY_PREDICT_WAIT_SECONDS", 30)
    monkeypatch.setattr(webapp, "_compute_legacy_predict", fake_compute())
    t0 = time.time()
    r = post_predict(c, hdr, **{"async": "1"})
    assert r.status_code == 202 and time.time() - t0 < 5 and 'id="job"' in r.get_data(as_text=True)
    jid = job_id(r); wait_done(c, jid)
    assert c.get(f"/predict/claim/{jid}").status_code == 302


def test_shipped_html_forms_send_async_but_the_noscript_fallback_does_not():
    for name in ("workstation.html", "index.html"):
        html = open(os.path.join(ROOT, "webapp/templates", name)).read()
        if name == "workstation.html":
            tpl = re.search(r"<template id=\"classic-form\">(.*?)</template>", html, re.S).group(1)
            fallback = re.search(r"<noscript>\s*<form(.*?)</noscript>", html, re.S).group(1)
            assert 'name="async" value="1"' in tpl and 'name="async"' not in fallback      # no-JS form cannot poll
        else:
            assert 'name="async" value="1"' in html


def test_programmatic_clients_keep_the_synchronous_contract_within_a_bounded_wait(client, monkeypatch):
    c, hdr = client
    monkeypatch.setattr(config, "LEGACY_PREDICT_WAIT_SECONDS", 30)
    monkeypatch.setattr(webapp, "_compute_legacy_predict", fake_compute())
    assert post_predict(c, hdr).status_code == 302                         # no async flag -> old behaviour
    def compute(*a):
        a[-1]["_payload"] = {"ticker": "AAPL"}; return "api_predict"
    monkeypatch.setattr(webapp, "_compute_api_predict", compute)
    assert c.post("/api/predict", json={"ticker": "AAPL"}).status_code == 200
    r = c.post("/api/predict", json={"ticker": "AAPL", "async": True})
    assert r.status_code == 202 and r.get_json()["job_id"]
