import math

import numpy as np
import pandas as pd
import pytest

import evaluation as ev
from tests.test_v17_evaluation import make_result


def frame(res_returns, base=100.0, extra_cols=None):
    """Frame whose model forecast is exactly `base` (point = base), so |return residual| == |res_returns|."""
    n = len(res_returns)
    base_arr = np.full(n, base)
    df = pd.DataFrame({"actual": base_arr * (1 + np.asarray(res_returns, float)), "base": base_arr, "m": base_arr},
                      index=[f"2025-01-01+{i:04d}" for i in range(n)])
    return df


def brute_q(past, level):
    m = len(past); k = math.ceil((m + 1) * level)
    return None if k > m else float(sorted(past)[k - 1])


RNG = np.random.default_rng(42)
NOISE = RNG.normal(0, 0.01, 260)


def test_interval_for_date_t_depends_only_on_residuals_before_t():
    f = frame(NOISE)
    base = ev.rolling_conformal(f, "m")
    j = 150                                                         # corrupt residual j and everything after it
    noisy = NOISE.copy(); noisy[j:] = 0.5
    changed = ev.rolling_conformal(frame(noisy), "m")
    by_date = lambda r: {s["date"]: (s["low"], s["high"]) for s in r["series"]}
    a, b = by_date(base), by_date(changed)
    dates = [f"2025-01-01+{i:04d}" for i in range(260)]
    for i in range(40, j + 1):                                       # dates <= j use only residuals < j  -> identical
        assert a[dates[i]] == b[dates[i]], i
    assert any(a[dates[i]] != b[dates[i]] for i in range(j + 1, 260))


def test_future_rows_cannot_change_any_historical_interval():
    short = ev.rolling_conformal(frame(NOISE[:180]), "m")
    long = ev.rolling_conformal(frame(np.r_[NOISE[:180], np.full(60, 0.3)]), "m")
    n = len(short["series"])
    assert [s["low"] for s in short["series"]] == [s["low"] for s in long["series"][:n]]


def test_displayed_historical_interval_is_the_evaluated_interval():
    f = frame(NOISE)
    r = ev.rolling_conformal(f, "m", level=0.8, window=100, min_cal=40)
    res = np.abs(NOISE)
    for s in r["series"][::17]:                                      # independent recomputation of the interval itself
        i = int(s["date"].split("+")[1])
        q = brute_q(res[max(0, i - 100):i], 0.8)
        assert s["q_return"] == pytest.approx(q) and s["low"] == pytest.approx(100 * (1 - q)) and s["high"] == pytest.approx(100 * (1 + q))
        assert s["covered"] == (s["low"] <= s["actual"] <= s["high"])
    assert r["observed_coverage"] == pytest.approx(np.mean([s["covered"] for s in r["series"]]))
    assert r["n_covered"] == sum(s["covered"] for s in r["series"]) and r["n_evaluated"] == len(r["series"])


def test_coverage_and_width_are_computed_correctly_on_a_known_pattern():
    res = np.array([0.01 * (1 + (i % 10) / 10) for i in range(120)])          # deterministic, cyclic
    sign = np.where(np.arange(120) % 3 == 0, -1, 1)
    r = ev.rolling_conformal(frame(res * sign), "m", level=0.8, window=1000, min_cal=40, min_eval=30)
    exp_cov, exp_w = [], []
    for i in range(40, 120):
        q = brute_q(res[:i], 0.8); exp_cov.append(res[i] <= q); exp_w.append(2 * q)
    assert r["observed_coverage"] == pytest.approx(np.mean(exp_cov)) and r["mean_relative_width"] == pytest.approx(np.mean(exp_w))
    assert r["mean_width_price"] == pytest.approx(np.mean([x["high"] - x["low"] for x in r["series"]]))


def test_insufficient_calibration_or_evaluation_data_is_explicit_and_has_no_numbers():
    r = ev.rolling_conformal(frame(NOISE[:60]), "m")                  # 60 residuals -> only 20 evaluable (<30)
    assert r["status"] == "insufficient_evaluation_data" and "Insufficient evaluation data" in r["message"] and "series" not in r
    fi = ev.forecast_interval(frame(NOISE[:60]), "m", 100.0, 100.0)
    assert fi["status"] == "insufficient_evaluation_data" and "low" not in fi and "high" not in fi
    assert ev.forecast_interval(frame(NOISE[:5]), "m", 100.0, 100.0)["status"] == "insufficient_evaluation_data"


def test_unsupported_level_is_never_silently_clamped_to_the_maximum():
    assert ev._conformal_q(np.arange(10.0), 0.95) is None            # rank ceil(11*.95)=11 > 10
    assert ev._conformal_q(np.arange(40.0), 0.8) == pytest.approx(brute_q(np.arange(40.0), 0.8))
    r = ev.rolling_conformal(frame(NOISE), "m", level=0.999, window=60, min_cal=40, min_eval=1)
    assert r["status"] != "ok" or r["n_unavailable_level_unsupported"] > 0


def test_nonfinite_residuals_are_dropped_counted_and_never_propagate():
    f = frame(NOISE[:200])
    f.iloc[10, f.columns.get_loc("m")] = np.nan
    f.iloc[20, f.columns.get_loc("actual")] = np.inf
    f.iloc[30, f.columns.get_loc("base")] = 0.0
    r = ev.rolling_conformal(f, "m")
    assert r["status"] == "ok" and r["n_nonfinite_dropped"] == 3 and r["n_residuals"] == 197
    assert all(np.isfinite([s["low"], s["high"], s["q_return"]]).all() for s in r["series"])


def test_nominal_coverage_is_never_presented_as_observed_coverage():
    r = ev.rolling_conformal(frame(NOISE), "m", level=0.8)
    assert r["nominal_coverage"] == 0.8 and r["observed_coverage"] != 0.8 and "coverage" not in r
    fi = ev.forecast_interval(frame(NOISE), "m", 100.0, 100.0)
    assert "observed_coverage" not in fi and fi["nominal_coverage"] == 0.8
    assert fi["evaluation"]["observed_coverage"] == r["observed_coverage"] and "does not validate this particular interval" in fi["reading_guide"]


def test_forecast_interval_uses_only_residuals_dated_before_the_forecast_and_the_same_procedure():
    f = frame(NOISE[:220])
    fi = ev.forecast_interval(f, "m", point_price=105.0, base_price=104.0, level=0.8, window=100)
    q = brute_q(np.abs(NOISE[:220])[-100:], 0.8)
    assert fi["q_return"] == pytest.approx(q) and fi["low"] == pytest.approx(105.0 - 104.0 * q) and fi["high"] == pytest.approx(105.0 + 104.0 * q)
    assert fi["calibration"]["end"] == "2025-01-01+0219" and fi["calibration"]["n"] == 100
    assert fi["evaluation"]["window"] == 100                                     # same procedure that was evaluated


def test_a_sudden_volatility_jump_shows_up_as_under_coverage_not_hidden():
    sig = np.r_[np.full(200, 0.005), np.full(60, 0.02)]
    res = np.random.default_rng(1).normal(0, sig)
    r = ev.rolling_conformal(frame(res), "m", level=0.8, window=120)
    post = [s["covered"] for s in r["series"] if int(s["date"].split("+")[1]) >= 200]
    assert np.mean(post) < 0.6 and r["observed_coverage"] < 0.8
    assert "No coverage guarantee" in r["caveat"]


def test_evaluation_pipeline_exposes_the_same_scored_intervals_and_counts_dropped_rows():
    res = make_result(n=340, n_test=150)
    out = ev.build_evaluation(res)
    ie = out["interval_evaluation"]
    assert ie["status"] == "ok" and ie["n_evaluated"] == 150 - 40 and out["sufficient"] and out["n_dropped_nonfinite"] == 0
    bad = make_result(n=340, n_test=150)
    bad["lr_eval"]["predictions"] = np.asarray(bad["lr_eval"]["predictions"], float).copy()
    bad["lr_eval"]["predictions"][5] = np.nan
    assert ev.build_evaluation(bad)["n_dropped_nonfinite"] == 1


def test_small_samples_report_insufficient_evaluation_data_instead_of_a_verdict():
    out = ev.build_evaluation(make_result(n=120, n_test=20))
    assert out["status"] == "insufficient_evaluation_data" and out["sufficient"] is False
    assert out["conclusion"].startswith("Insufficient evaluation data")
