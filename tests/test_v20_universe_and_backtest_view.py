"""Two V20 additions that belong with earlier suites, kept in a NEW file so the V19 test files stay byte-identical."""
import os

import pytest

import config
import universe_eval as ue
from tests.test_v19_universe import universe                      # noqa: F401  (fixture)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = open(os.path.join(ROOT, "webapp/static/ws.js"), encoding="utf-8").read()


def test_summary_separates_nominal_wins_from_statistically_supported_ones(universe):
    fetch, _ = universe
    out = ue.evaluate_universe(["A", "B", "C"], epochs=1, fetch=fetch)
    for m, v in out["summary"]["per_model"].items():
        assert v["n_significantly_better"] <= v["n_beat_naive"] <= v["n_instruments"]
        assert v["n_significantly_better"] + v["n_significantly_worse"] <= v["n_instruments"]
    row = out["rows"][0]["models"]["linear_regression"]
    assert "rmse_improvement_ci99" in row and "mase" in row


def test_backtest_view_explains_a_cash_only_or_low_exposure_result_instead_of_implying_skill():
    assert "No trades were made." in JS and "only reflects how the market moved" in JS and "not evidence of skill" in JS
    assert "r.exposure_pct < 50" in JS and "will beat Buy & Hold in a falling market" in JS
