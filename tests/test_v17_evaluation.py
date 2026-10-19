import numpy as np
import pandas as pd
import pytest

import evaluation as ev
from tests.v17_helpers import fixture_bars


def make_result(n=300, n_test=100, seed=1, lr_noise=0.01, ls_noise=0.01):
    df = fixture_bars(n, seed=seed)
    close = df["Close"].to_numpy()
    rng = np.random.default_rng(seed + 100)
    idx = list(range(n - n_test, n))
    dates = [df.index[i] for i in idx]
    act = close[idx]
    base = close[[i - 1 for i in idx]]
    lr = base * (1 + rng.normal(0, lr_noise, n_test))
    ls = base * (1 + rng.normal(0, ls_noise, n_test))
    return {"df": df, "lr_dates_test": dates, "y_test": act, "lr_eval": {"predictions": lr},
            "lstm_dates_test": dates, "lstm_eval": {"actual": act, "predictions": ls}}


def test_base_is_previous_close_and_naive_rmse_matches_manual():
    r = make_result()
    f = ev.aligned_frame(r)
    close = r["df"]["Close"]
    first = f.index[0]
    pos = close.index.get_loc(pd.Timestamp(first))
    assert f["base"].iloc[0] == pytest.approx(close.iloc[pos - 1]) and f["actual"].iloc[0] == pytest.approx(close.iloc[pos])
    res = ev.evaluate_frame(f)
    naive = next(m for m in res["models"] if m["model"] == "naive")
    manual = np.sqrt(np.mean((f["actual"] - f["base"]) ** 2))
    assert naive["rmse"] == pytest.approx(manual) and res["n_obs"] == 100 and res["out_of_sample"]


def test_misaligned_targets_are_detected_not_silently_scored():
    r = make_result()
    r["y_test"] = np.roll(r["y_test"], 1)
    with pytest.raises(ValueError, match="misaligned"):
        ev.aligned_frame(r)


def test_drift_baseline_uses_only_data_before_the_test_window():
    r = make_result()
    mu1 = ev.aligned_frame(r)["drift"].attrs if False else ev.aligned_frame(r).attrs["drift_mean_return"]
    r2 = make_result()
    first = r2["lr_dates_test"][0]
    r2["df"].loc[r2["df"].index > first, "Close"] *= 5          # corrupt the future only
    # keep targets consistent with the corrupted df so alignment passes
    close = r2["df"]["Close"]
    r2["y_test"] = close.loc[r2["lr_dates_test"]].to_numpy(); r2["lstm_eval"]["actual"] = r2["y_test"]
    mu2 = ev.aligned_frame(r2).attrs["drift_mean_return"]
    assert mu1 == pytest.approx(mu2)


def test_a_model_that_is_just_the_naive_forecast_shows_no_improvement():
    r = make_result(lr_noise=0.0, ls_noise=0.0)
    res = ev.build_evaluation(r)
    lr = next(m for m in res["models"] if m["model"] == "linear_regression")
    assert lr["rmse_improvement_vs_naive_pct"] == pytest.approx(0, abs=1e-9)
    assert lr["directional_accuracy_pct"] is None and "no directional call" in lr["unavailable"]["directional_accuracy_pct"]


def test_model_with_real_skill_beats_naive_and_is_reported_as_such():
    r = make_result(lr_noise=0.05)
    close = r["df"]["Close"]
    idx = [close.index.get_loc(d) for d in r["lr_dates_test"]]
    base = close.iloc[[i - 1 for i in idx]].to_numpy()
    skilled = base + 0.8 * (close.iloc[idx].to_numpy() - base)     # test fixture: deliberately informed model
    r["lr_eval"]["predictions"] = skilled
    out = ev.build_evaluation(r)
    lr = next(m for m in out["models"] if m["model"] == "linear_regression")
    assert lr["rmse_improvement_vs_naive_pct"] > 50 and lr["diebold_mariano_vs_naive"]["z"] > 3
    assert "linear_regression: beats" in out["conclusion"]


def test_undefined_metrics_are_none_with_reason_never_zero():
    r = make_result(n=60, n_test=8)
    res = ev.evaluate_frame(ev.aligned_frame(r))
    for m in res["models"]:
        if m["model"] in ("naive",):
            continue
        assert m["directional_accuracy_pct"] is None or m["directional_n"] >= 10
        if m["directional_accuracy_pct"] is None:
            assert "directional_accuracy_pct" in m["unavailable"]
        assert m["mape"] is None and "mape" in m["unavailable"]       # fewer than 10 obs


def test_mape_is_undefined_for_zero_targets():
    assert ev._mape(np.zeros(50), np.ones(50)) is None


def test_flat_actual_days_are_excluded_from_direction():
    a = np.array([10.0] * 12 + [11.0] * 12); base = np.r_[10.0, a[:-1]]
    pred = base + 0.5
    acc, n, why = ev.directional_accuracy_vs_base(a, pred, base)
    assert n == 1 and acc is None            # only one non-flat day -> below minimum, so undefined


def test_interval_refuses_when_sample_too_small():
    f = ev.aligned_frame(make_result(n_test=40))
    out = ev.conformal_interval(f, "lr_lstm_ensemble", 100.0, 99.0)
    assert out["status"] == "insufficient_evaluation_data" and "Insufficient evaluation data" in out["message"]

# (V18: the two V17 interval tests that asserted the old mixed-quantile behaviour were replaced by tests/test_v18_conformal.py)
