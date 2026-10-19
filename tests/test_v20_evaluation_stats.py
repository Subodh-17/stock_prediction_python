import numpy as np
import pandas as pd
import pytest

import evaluation as ev
from tests.test_v17_evaluation import make_result


def rows(out):
    return {r["model"]: r for r in out["models"]}


# ---- MASE
def test_mase_matches_a_manual_calculation_and_uses_only_training_period_data():
    r = make_result(n=340, n_test=120)
    f = ev.aligned_frame(r)
    close = r["df"]["Close"]; first = pd.Timestamp(f.index[0])
    scale = close[close.index < first].diff().abs().dropna().mean()
    assert f.attrs["naive_insample_mae"] == pytest.approx(scale)
    out = ev.evaluate_frame(f)
    lr = rows(out)["linear_regression"]
    assert lr["mase"] == pytest.approx(lr["mae"] / scale)


def test_mase_scale_cannot_be_influenced_by_future_prices():
    r = make_result(n=340, n_test=120)
    base = ev.aligned_frame(r).attrs["naive_insample_mae"]
    r2 = make_result(n=340, n_test=120)
    first = r2["lr_dates_test"][0]
    r2["df"].loc[r2["df"].index >= first, "Close"] *= 3          # corrupt the test window and everything after it
    r2["y_test"] = r2["df"]["Close"].loc[r2["lr_dates_test"]].to_numpy(); r2["lstm_eval"]["actual"] = r2["y_test"]
    assert ev.aligned_frame(r2).attrs["naive_insample_mae"] == pytest.approx(base)


def test_mase_is_unavailable_with_a_reason_when_the_training_period_is_too_short():
    out = ev.evaluate_frame(ev.aligned_frame(make_result(n=60, n_test=40)))
    lr = rows(out)["linear_regression"]
    assert lr["mase"] is None and "mase" in lr["unavailable"] and "training-period" in lr["unavailable"]["mase"]


# ---- bootstrap intervals
def test_bootstrap_is_deterministic_and_paired_errors_identical_to_naive_give_a_zero_interval():
    rng = np.random.default_rng(1)
    e_n = rng.normal(0, 1, 120); e_m = e_n * 0.8 + rng.normal(0, 0.2, 120)
    a = ev.bootstrap_rmse_improvement_cis(e_m, e_n); b = ev.bootstrap_rmse_improvement_cis(e_m, e_n)
    assert a == b and set(a) == {0.95, 0.99} and a[0.99][0] <= a[0.95][0] <= a[0.95][1] <= a[0.99][1]      # 99% interval is wider
    point = (1 - np.sqrt(np.mean(e_m ** 2)) / np.sqrt(np.mean(e_n ** 2))) * 100
    assert a[0.95][0] <= point <= a[0.95][1]
    same = ev.bootstrap_rmse_improvement_cis(e_n, e_n)
    assert same[0.95] == [0.0, 0.0] and same[0.99] == [0.0, 0.0]


def test_bootstrap_refuses_small_samples_and_zero_naive_error():
    rng = np.random.default_rng(2)
    assert ev.bootstrap_rmse_improvement_cis(rng.normal(size=ev.MIN_OBS_EVAL - 1), rng.normal(size=ev.MIN_OBS_EVAL - 1)) is None
    assert ev.bootstrap_rmse_improvement_cis(rng.normal(size=60), np.zeros(60)) is None
    assert ev.bootstrap_directional_ci(np.ones(10)) is None


def test_a_genuinely_better_model_has_a_positive_interval_and_a_worse_one_a_negative_interval():
    rng = np.random.default_rng(3)
    e_n = rng.normal(0, 1, 200)
    better = ev.bootstrap_rmse_improvement_cis(e_n * 0.5, e_n); worse = ev.bootstrap_rmse_improvement_cis(e_n * 2.0, e_n)
    assert better[0.99][0] > 0 and worse[0.99][1] < 0


def test_directional_ci_brackets_the_observed_accuracy():
    rng = np.random.default_rng(4)
    hits = (rng.random(150) < 0.6).astype(float)
    lo, hi = ev.bootstrap_directional_ci(hits)
    assert 0 <= lo <= hits.mean() * 100 <= hi <= 100


def test_calibration_regression_ci99_is_far_less_prone_to_false_skill_claims_than_ci95():
    """Regression for the V20 finding: under NO true skill the 95% bootstrap interval says 'beats naive' too often. Seeded replay."""
    rng = np.random.default_rng(777)
    n, reps, b95, b99 = 120, 150, 0, 0
    for _ in range(reps):
        vol = np.empty(n); vol[0] = 1.0
        for i in range(1, n):
            vol[i] = np.exp(0.8 * np.log(vol[i - 1]) + 0.3 * rng.normal())
        e_n, e_m = rng.normal(0, 1, n) * vol, rng.normal(0, 1, n) * vol
        c = ev.bootstrap_rmse_improvement_cis(e_m, e_n, B=300, seed=int(rng.integers(1e9)))
        b95 += c[0.95][0] > 0; b99 += c[0.99][0] > 0
    assert b99 / reps <= 0.06                      # what the verdict actually uses
    assert b95 > b99                               # documents why the 95% interval is not used for the verdict


# ---- verdict wording and metadata
def test_verdict_requires_the_99_percent_interval_to_exclude_zero():
    skilled = make_result(n=340, n_test=120, lr_noise=0.05)
    close = skilled["df"]["Close"]; idx = [close.index.get_loc(d) for d in skilled["lr_dates_test"]]
    base = close.iloc[[i - 1 for i in idx]].to_numpy(); act = close.iloc[idx].to_numpy()
    skilled["lr_eval"]["predictions"] = base + 0.8 * (act - base)
    out = ev.build_evaluation(skilled)
    assert "linear_regression: beats the naive last-close baseline" in out["conclusion"] and "99% CI" in out["conclusion"]
    noisy = ev.build_evaluation(make_result(n=340, n_test=120))
    assert "does NOT beat" in noisy["conclusion"] and "WORSE" in noisy["conclusion"]
    equal = ev.build_evaluation(make_result(n=340, n_test=120, lr_noise=0.0, ls_noise=0.0))
    assert "not distinguishable" in equal["conclusion"]


def test_evaluation_declares_a_single_one_day_horizon_and_documents_its_scaled_metric_and_intervals():
    out = ev.build_evaluation(make_result(n=340, n_test=120))
    assert out["horizon"]["days"] == 1 and "NEXT trading day" in out["horizon"]["definition"]
    assert "do not produce multi-day forecasts" in out["horizon"]["definition"]
    assert "MASE" in out["mase_note"] and "99%" in out["ci_note"] and "anti-conservative" in out["ci_note"]
    for m in out["models"]:
        if m["model"] != "naive":
            assert m["rmse_improvement_ci99"] is not None and m["mase"] is not None
    assert out["models"][0]["rmse_improvement_ci95"] is None            # the baseline is not compared with itself
