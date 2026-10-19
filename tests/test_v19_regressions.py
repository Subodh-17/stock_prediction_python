"""Regression tests for defects reproduced against V18 (see docs/V19_FINAL_AUDIT.md section 2)."""
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "webapp"), ROOT]

import checkpoints as ck
import config
import evaluation as ev
import market_data as md
from tests.test_v17_evaluation import make_result
from tests.test_v17_market_data import Fake, raw
from tests.v17_helpers import fixture_bars


# ---- A: a stale real-time quote must stay stale after caching (and therefore stay blocked for paper orders)
def stale_rt_provider():
    return Fake("rt", raw=raw(ts=datetime.now(timezone.utc) - timedelta(minutes=10), dtype="realtime", delay=None), realtime=True)


def test_cached_stale_realtime_quote_stays_stale_not_unverified():
    svc = md.MarketDataService([stale_rt_provider()])
    first = svc.get_quote("NSE:RELIANCE", force=True)
    cached = svc.get_quote("NSE:RELIANCE")
    assert first.freshness == "stale" and cached.from_cache
    assert cached.freshness == "stale" and cached.is_stale and not cached.realtime_eligible
    assert cached.provider_entitled_realtime is True


def test_cached_live_quote_ages_into_stale_instead_of_staying_live(monkeypatch):
    fresh = Fake("rt", raw=raw(ts=datetime.now(timezone.utc) - timedelta(seconds=2), dtype="realtime", delay=None), realtime=True)
    svc = md.MarketDataService([fresh])
    assert svc.get_quote("NSE:TCS", force=True).freshness == "live"
    monkeypatch.setattr(config, "LIVE_MAX_AGE_SECONDS", 0.5)
    time.sleep(0.7)
    again = svc.get_quote("NSE:TCS")
    assert again.from_cache and again.freshness == "stale" and not again.realtime_eligible


def test_paper_order_is_refused_for_a_cached_stale_quote(tmp_path, monkeypatch):
    from tests.test_v17_webapp import webapp, api_v2
    monkeypatch.setattr(config, "PAPER_PORTFOLIO_PATH", str(tmp_path / "p.json"))
    monkeypatch.setattr(config, "TRADE_LOG_PATH", str(tmp_path / "l.csv"))
    monkeypatch.setattr(config, "WEBAPP_EXPENSIVE_RPM", 1000)
    api_v2._hits.clear()
    svc = md.MarketDataService([stale_rt_provider()])
    monkeypatch.setattr(md, "_service", svc)
    c = webapp.app.test_client()
    hdr = {"X-CSRF-Token": c.get("/api/v2/meta").get_json()["csrf_token"]}
    svc.get_quote("NSE:RELIANCE", force=True)                       # populates the cache with the stale quote
    monkeypatch.setattr(svc, "get_quote", lambda *a, force=False, **k: md.MarketDataService.get_quote(svc, *a, force=False, **k))
    r = c.post("/api/v2/paper/order", json={"symbol": "NSE:RELIANCE", "side": "BUY", "quantity": 1, "confirm": True}, headers=hdr)
    assert r.status_code == 409 and r.get_json()["error"] == "no_fresh_quote"
    assert c.get("/api/v2/portfolio").get_json()["positions"] == []


# ---- B: a failing history provider is not re-hit on every refresh
def test_history_failures_are_negatively_cached_and_retried_after_the_ttl(monkeypatch):
    import data_sources.yfinance_source as ys
    from data_sources.base import DataSourceError
    calls = {"n": 0}
    def failing(*a, **k):
        calls["n"] += 1
        raise DataSourceError("HTTP 500 upstream")
    monkeypatch.setattr(ys, "fetch", failing)
    monkeypatch.setattr(config, "HISTORY_FAILURE_TTL_SECONDS", 0.3)
    svc = md.MarketDataService([])
    for _ in range(6):
        with pytest.raises(md.ProviderError):
            svc.get_history("AAPL", "1y")
    assert calls["n"] <= 3 and svc.failure_cache_hits >= 5          # initial attempt (with its bounded retries), then no upstream calls
    before = calls["n"]
    time.sleep(0.4)
    with pytest.raises(md.ProviderError):
        svc.get_history("AAPL", "1y")
    assert calls["n"] > before                                       # TTL expired -> tried again


def test_a_later_success_clears_the_way_and_validation_errors_are_not_cached(monkeypatch):
    import data_sources.yfinance_source as ys
    state = {"ok": False, "n": 0}
    def flaky(*a, **k):
        state["n"] += 1
        if not state["ok"]:
            from data_sources.base import DataSourceError
            raise DataSourceError("HTTP 500")
        return fixture_bars(60)
    monkeypatch.setattr(ys, "fetch", flaky)
    monkeypatch.setattr(config, "HISTORY_FAILURE_TTL_SECONDS", 0.2)
    svc = md.MarketDataService([])
    with pytest.raises(md.ProviderError):
        svc.get_history("MSFT", "1y")
    state["ok"] = True; time.sleep(0.3)
    assert svc.get_history("MSFT", "1y")["rows"] == 60
    with pytest.raises(ValueError):
        svc.get_history("MSFT", "bad period")


# ---- C: missing bars are not scored as next-day targets
def _result_with_missing_bars(first=300, count=12):
    r = make_result(n=340, n_test=120)
    df = r["df"]; drop = df.index[first:first + count]
    keep = [i for i, d in enumerate(r["lr_dates_test"]) if d not in drop]
    r2 = dict(r); r2["df"] = df.drop(drop)
    r2["lr_dates_test"] = [r["lr_dates_test"][i] for i in keep]; r2["lstm_dates_test"] = r2["lr_dates_test"]
    r2["y_test"] = r["y_test"][keep]; r2["lr_eval"] = {"predictions": r["lr_eval"]["predictions"][keep]}
    r2["lstm_eval"] = {"actual": r["lstm_eval"]["actual"][keep], "predictions": r["lstm_eval"]["predictions"][keep]}
    return r2


def test_a_multi_week_data_gap_is_excluded_counted_and_reported():
    f = ev.aligned_frame(_result_with_missing_bars())
    assert f.attrs["n_dropped_gap"] == 1                                # exactly the one target that follows the hole
    close = _result_with_missing_bars()["df"]["Close"]
    for d in f.index:
        i = close.index.get_loc(pd.Timestamp(d))
        assert (close.index[i] - close.index[i - 1]).days <= ev.MAX_GAP_DAYS
    out = ev.build_evaluation(_result_with_missing_bars())
    assert out["n_dropped_gap"] == 1 and out["max_gap_days"] == ev.MAX_GAP_DAYS


def test_normal_weekends_and_holiday_clusters_are_not_dropped():
    out = ev.build_evaluation(make_result(n=340, n_test=120))           # business-day index: weekends only (gap 3 days)
    assert out["n_dropped_gap"] == 0 and out["n_obs"] == 120


# ---- D: an LLM cannot influence the numerical forecast without an explicit acknowledgement
def _weights(**env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("LLM_PANEL")}
    e.update(env)
    out = subprocess.run([sys.executable, "-c", "import config; print(config.ENSEMBLE_WEIGHTS['llm_panel'], '|', config.LLM_WEIGHT_NOTE)"],
                         cwd=ROOT, capture_output=True, text=True, env=e, timeout=120).stdout.strip()
    w, note = out.split(" | ", 1)
    return float(w), note


def test_llm_weight_defaults_to_zero_and_a_requested_weight_is_ignored_without_evidence_ack():
    assert _weights() == (0.0, "None")
    w, note = _weights(LLM_PANEL_WEIGHT="0.3")
    assert w == 0.0 and "IGNORED" in note
    w, note = _weights(LLM_PANEL_WEIGHT="0.3", LLM_PANEL_WEIGHT_EVIDENCE_ACK="yes")
    assert w == 0.0                                                      # a vague "yes" is not the acknowledgement


def test_llm_weight_requires_the_exact_acknowledgement_and_is_then_flagged_as_unvalidated():
    w, note = _weights(LLM_PANEL_WEIGHT="0.3", LLM_PANEL_WEIGHT_EVIDENCE_ACK="I_HAVE_OUT_OF_SAMPLE_EVIDENCE")
    assert w == 0.3 and "NOT validated" in note


# ---- E: checkpoints saved under an incompatible dependency major version are rejected
META = {"manifest_version": 2, "train_end": "2024-01-01", "model_type": "lstm", "model_impl_version": ck.MODEL_IMPL_VERSION,
        "feature_version": ck.FEATURE_VERSION, "feature_names": ["Close_return"], "forecast_horizon_days": 1,
        "scaler_config": {"n_features": 1}, "package_versions": {"tensorflow": "2.21.0", "keras": "3.15.1", "numpy": "2.5.4"}}
NOW = {"tensorflow": "2.21.0", "keras": "3.15.1", "numpy": "2.5.4"}


def problems(meta=META, now=NOW):
    return ck.validate_provenance(meta, require=True, first_eval_date="2025-01-01", current_versions=now)


def test_same_or_minor_dependency_versions_are_accepted_and_major_changes_rejected():
    assert problems() == []
    assert problems(now={"tensorflow": "2.22.1", "keras": "3.16.0", "numpy": "2.6.0"}) == []
    p = problems(now={"tensorflow": "3.0.0", "keras": "3.15.1", "numpy": "2.5.4"})
    assert len(p) == 1 and "tensorflow" in p[0] and "different major" in p[0]
    assert any("keras" in x for x in problems(now={**NOW, "keras": "4.0.0"}))
    assert any("numpy" in x for x in problems(now={**NOW, "numpy": "3.0.0"}))


def test_manifest_without_dependency_versions_cannot_be_proven_compatible():
    m = {k: v for k, v in META.items() if k != "package_versions"}
    assert any("no dependency versions" in x for x in problems(meta=m))


def test_load_checkpoint_refuses_incompatible_dependency_versions_end_to_end(tmp_path, monkeypatch):
    from tests.test_v18_checkpoints import save, load, DF
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path / "ck")); monkeypatch.setattr(config, "TRAIN_TEST_SPLIT", 0.8)
    save()
    assert load()[0].is_fitted
    import reproducibility
    real = reproducibility.get_package_versions()
    monkeypatch.setattr(reproducibility, "get_package_versions", lambda: {**real, "tensorflow": "9.9.9", "keras": "9.0.0"})
    with pytest.raises(ck.CheckpointProvenanceError, match="different major version"):
        load()
