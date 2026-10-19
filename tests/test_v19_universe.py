import pytest

import config
import market_data as md
import universe_eval as ue
from tests.v17_helpers import fixture_bars


def test_sign_test_is_exact():
    assert ue.sign_test_p(8, 10) == pytest.approx(2 * (1 + 10 + 45) / 1024)
    assert ue.sign_test_p(5, 10) == pytest.approx(1.0) and ue.sign_test_p(0, 0) is None
    assert ue.sign_test_p(0, 6) == pytest.approx(2 / 64) and ue.sign_test_p(6, 6) == pytest.approx(2 / 64)


@pytest.fixture
def universe(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path / "ck"))
    calls = []
    def fetch(sym, period):
        calls.append(sym)
        if sym == "BAD":
            raise md.ProviderError("HTTP 500 from provider key=AIzaSyLEAK0000000000000000000000000")
        if sym == "SHORT":
            return fixture_bars(40, seed=1)
        return fixture_bars(420, seed={"A": 11, "B": 12, "C": 13}[sym])
    return fetch, calls


def test_every_requested_symbol_is_reported_in_order_including_failures_and_nothing_is_retried(universe, monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "AIzaSyLEAK0000000000000000000000000")
    fetch, calls = universe
    syms = ["A", "BAD", "B", "SHORT", "C"]
    out = ue.evaluate_universe(syms, epochs=1, fetch=fetch)
    assert [r["symbol"] for r in out["rows"]] == syms and out["symbols_requested"] == syms
    assert calls == syms                                              # each fetched exactly once: no re-selection, no retry
    status = {r["symbol"]: r["status"] for r in out["rows"]}
    assert status["A"] == status["B"] == status["C"] == "ok" and status["BAD"] == status["SHORT"] == "failed"
    s = out["summary"]
    assert s["requested"] == 5 and s["evaluated_ok"] == 3 and s["failed"] == 2
    bad = next(r for r in out["rows"] if r["symbol"] == "BAD")
    assert "HTTP 500" in bad["error"] and "AIzaSyLEAK" not in bad["error"]          # failures are shown, secrets are not


def test_summary_counts_every_evaluated_instrument_for_every_model_and_states_its_limits(universe):
    fetch, _ = universe
    out = ue.evaluate_universe(["A", "B", "C"], epochs=1, fetch=fetch)
    for m, v in out["summary"]["per_model"].items():
        assert v["n_instruments"] == 3 and v["n_beat_naive"] + v["n_not_beat_naive"] == 3
        assert v["worst_pct"] <= v["median_rmse_improvement_pct"] <= v["best_pct"]
        assert 0 < v["sign_test_p_two_sided"] <= 1
    guide = out["summary"]["reading_guide"]
    assert "not independent" in guide and "not a trading edge" in guide
    for r in out["rows"]:
        assert r["n_obs"] >= 30 and r["period"]["start"] < r["period"]["end"] and r["epochs"] == 1
