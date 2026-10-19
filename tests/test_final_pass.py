"""
tests/test_final_pass.py
-------------------------
Regression tests for the issues found in the final production-readiness
pass. ADDITIVE -- no existing test was removed, weakened, or skipped to
accommodate any of these.

Covers, in order:
  1. LR test/fold dates are labeled by TARGET date, so the LR and LSTM
     series are directly comparable (the one-day misalignment bug)
  2. Future data cannot influence historical predictions (leakage proof)
  3. Checkpoint compatibility: a train/test-split mismatch is refused
  4. Ensemble degradation: one failed model does not corrupt the blend
  5. Export robustness: a missing metric doesn't 500 the CSV download
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

import config
import data_loader
import ensemble
import validation as validation_mod


def _synthetic_df(n=260, seed=7, start=120.0):
    """Deterministic OHLCV data with a real trend, so leakage shocks show up."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    returns = rng.normal(loc=0.0006, scale=0.014, size=n)
    close = start * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.004, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.004, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 4_000_000, n),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# 1. LR dates label the TARGET day, not the feature day
# ---------------------------------------------------------------------------
def test_lr_test_dates_label_the_day_the_target_was_realized():
    """
    Each returned test date must be the day its y_test value actually
    happened on. Before the fix these were FEATURE dates -- one trading day
    early -- because add_technical_features sets Target = Close.shift(-1).
    """
    df = _synthetic_df()
    _, _, _, y_test, _, dates_test = data_loader.prepare_linear_regression_data(df)

    assert len(dates_test) == len(y_test)
    for date, target in zip(dates_test, y_test):
        assert date in df.index, f"{date} is not a real trading day in the source data"
        assert target == pytest.approx(float(df.loc[date, "Close"])), (
            f"y_test value for {date} is {target} but that day's actual close is "
            f"{float(df.loc[date, 'Close'])} -- the series is misaligned by a day"
        )


def test_lr_and_lstm_test_series_agree_on_the_actual_price_per_date():
    """
    The ensemble backtest, the CSV/PDF export table and the comparison chart
    all join the LR and LSTM series on date. That join is only meaningful if
    both label a given date with the SAME actual close. Before the fix, LR
    reported close(d+1) on date d while the LSTM reported close(d).
    """
    df = _synthetic_df()
    _, _, _, lr_y, _, lr_dates = data_loader.prepare_linear_regression_data(df)
    _, _, _, lstm_y, scaler, lstm_dates, base_prices = \
        data_loader.prepare_lstm_training_data(df, target_mode="return")

    lstm_returns = scaler.inverse_transform(np.asarray(lstm_y).reshape(-1, 1)).flatten()
    lstm_actual = np.asarray(base_prices, dtype=float) * (1 + lstm_returns)

    lr_by_date = {str(d)[:10]: float(v) for d, v in zip(lr_dates, lr_y)}
    lstm_by_date = {str(d)[:10]: float(v) for d, v in zip(lstm_dates, lstm_actual)}
    common = sorted(set(lr_by_date) & set(lstm_by_date))
    assert len(common) > 10, "expected a substantial overlap to test the join on"

    for d in common:
        assert lr_by_date[d] == pytest.approx(lstm_by_date[d], rel=1e-9), (
            f"on {d} the LR series says the actual close was {lr_by_date[d]} but the "
            f"LSTM series says {lstm_by_date[d]} -- the two are offset by a day"
        )


def test_lr_walk_forward_fold_boundaries_are_real_target_dates():
    """
    Fold test_end_date feeds backtest._compute_historical_dynamic_weights'
    "only folds that concluded strictly BEFORE this day" filter, so it must
    be the date the fold's last target was actually realized -- otherwise
    that filter admits a fold that had already seen the day being weighted.
    """
    df = _synthetic_df()
    result = validation_mod.walk_forward_validate_lr(df, n_splits=3)
    assert result.n_folds >= 2

    featured = data_loader.add_technical_features(df)
    target_dates = {str(d)[:10] for d in data_loader.lr_target_dates(df, featured.index)}
    for fold in result.folds:
        assert fold.test_end_date in target_dates
        assert fold.test_start_date in target_dates
        assert fold.test_start_date <= fold.test_end_date
    # Expanding window: each fold starts where the previous one ended.
    for earlier, later in zip(result.folds, result.folds[1:]):
        assert later.test_start_date > earlier.test_end_date
        assert later.train_size > earlier.train_size


# ---------------------------------------------------------------------------
# 2. Future data cannot influence historical predictions
# ---------------------------------------------------------------------------
def test_future_data_cannot_change_an_earlier_walk_forward_folds_results():
    """
    The direct leakage proof: take a walk-forward run, then violently shock
    every price AFTER the first fold's test period concluded, and re-run.
    Fold 1's metrics must be bit-for-bit identical -- it could not have known
    any of that data at the time, so nothing about it may reach fold 1.

    Fold boundaries are index-based and both DataFrames are the same length,
    so the folds line up; the LR model is deterministic, so any difference
    here would be genuine leakage rather than training noise.
    """
    df = _synthetic_df()
    baseline = validation_mod.walk_forward_validate_lr(df, n_splits=3)
    assert baseline.n_folds >= 2, "need multiple folds for this test to mean anything"

    fold1 = baseline.folds[0]
    # Shock everything strictly after fold 1's last realized target date.
    shock_start = df.index.get_loc(pd.Timestamp(fold1.test_end_date)) + 1
    shocked = df.copy()
    shocked.iloc[shock_start:, shocked.columns.get_indexer(["Open", "High", "Low", "Close"])] *= 5.0

    after = validation_mod.walk_forward_validate_lr(shocked, n_splits=3)
    shocked_fold1 = after.folds[0]

    assert shocked_fold1.test_end_date == fold1.test_end_date
    assert shocked_fold1.train_size == fold1.train_size
    assert shocked_fold1.rmse == pytest.approx(fold1.rmse, rel=1e-12, abs=1e-12)
    assert shocked_fold1.mae == pytest.approx(fold1.mae, rel=1e-12, abs=1e-12)
    assert shocked_fold1.directional_accuracy_pct == pytest.approx(
        fold1.directional_accuracy_pct, rel=1e-12, abs=1e-12
    )
    # Sanity check that the shock was actually large enough to have shown up
    # if any of it HAD leaked -- otherwise this test could pass vacuously.
    assert after.folds[-1].rmse != pytest.approx(baseline.folds[-1].rmse, rel=1e-6)


def test_lr_feature_row_is_unaffected_by_later_prices():
    """
    Every LR feature is backward-looking (lags, rolling means, rolling std),
    so a given day's feature row must not move when later days change.
    """
    df = _synthetic_df(n=80)
    featured = data_loader.add_technical_features(df)
    # Pick a real date from the FEATURED frame and index by date throughout:
    # add_technical_features drops leading rows for the 10-day moving average,
    # so featured.iloc[k] is not df.iloc[k].
    target_date = featured.index[40]
    row_before = featured.loc[target_date, data_loader.LR_FEATURE_COLUMNS].to_numpy(dtype=float)

    shocked = df.copy()
    shock_from = df.index.get_loc(target_date) + 1
    shocked.iloc[shock_from:, shocked.columns.get_indexer(["Open", "High", "Low", "Close"])] *= 7.0
    row_after = data_loader.add_technical_features(shocked).loc[
        target_date, data_loader.LR_FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    np.testing.assert_allclose(row_before, row_after, rtol=1e-12, atol=1e-12)


def test_lstm_scaler_parameters_ignore_the_test_period_entirely():
    """
    The MinMaxScaler must be fit on training rows only. Shocking the test
    rows must not move its fitted min/scale, nor any training sequence.
    """
    df = _synthetic_df(n=200)
    close = df["Close"].values.reshape(-1, 1)
    split = int(len(close) * 0.8)

    X_train_a, _, _, _, scaler_a = data_loader.build_lstm_sequences_leakage_safe(
        close[:split], close[split:], lookback=20
    )
    shocked_test = close[split:] * 9.0
    X_train_b, _, _, _, scaler_b = data_loader.build_lstm_sequences_leakage_safe(
        close[:split], shocked_test, lookback=20
    )

    np.testing.assert_allclose(scaler_a.data_min_, scaler_b.data_min_, rtol=0, atol=0)
    np.testing.assert_allclose(scaler_a.data_max_, scaler_b.data_max_, rtol=0, atol=0)
    np.testing.assert_allclose(X_train_a, X_train_b, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 3. Checkpoint compatibility
# ---------------------------------------------------------------------------
def test_checkpoint_with_a_different_train_test_split_is_refused(tmp_path, monkeypatch):
    """
    pipeline.py rebuilds the test window at the CURRENT split when reusing a
    checkpoint. If the checkpoint was trained under a larger split, that
    rebuilt "test" window starts inside rows the model was trained on, and
    lstm_eval would report in-sample performance as held-out. The metadata
    to catch this was always saved -- it just wasn't compared.
    """
    import checkpoints as checkpoints_mod
    from models.lstm_model import LSTMPredictor

    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path / "ckpts"))
    monkeypatch.setattr(checkpoints_mod.config, "CHECKPOINT_DIR", str(tmp_path / "ckpts"))
    monkeypatch.setattr(config, "TRAIN_TEST_SPLIT", 0.8)
    monkeypatch.setattr(checkpoints_mod.config, "TRAIN_TEST_SPLIT", 0.8)

    df = _synthetic_df(n=150)
    X_train, _, y_train, _, scaler, _, _ = data_loader.prepare_lstm_training_data(
        df, target_mode="return", lookback=10, split_ratio=0.8
    )
    model = LSTMPredictor(lookback=10)
    model.train(X_train, y_train, epochs=1, verbose=0)
    checkpoints_mod.save_checkpoint(
        "split-test", model, scaler, ticker="TESTX", period="1y",
        source="synthetic", n_train_rows=len(X_train), target_mode="return",
    )

    # Same everything -- loads fine.
    loaded, _, _ = checkpoints_mod.load_checkpoint(
        "split-test", ticker="TESTX", lookback=10, target_mode="return",
        train_test_split=0.8,
    )
    assert loaded.is_fitted

    # Different split -- must be refused, not silently reused.
    with pytest.raises(checkpoints_mod.CheckpointMismatchError) as exc:
        checkpoints_mod.load_checkpoint(
            "split-test", ticker="TESTX", lookback=10, target_mode="return",
            train_test_split=0.6,
        )
    assert "train_test_split" in str(exc.value)


# ---------------------------------------------------------------------------
# 4. Ensemble degradation and determinism
# ---------------------------------------------------------------------------
def test_healthy_two_model_blend_is_unchanged_by_the_degradation_path():
    """The normal case must behave exactly as it always did."""
    result = ensemble.compute_ensemble_prediction(
        lr_price=101.0, lstm_price=103.0, llm_panel_summary=None, current_price=100.0
    )
    w = config.ENSEMBLE_WEIGHTS
    remaining = w["linear_regression"] + w["lstm"]
    expected = (w["linear_regression"] / remaining) * 101.0 + (w["lstm"] / remaining) * 103.0
    assert result["final_predicted_price"] == pytest.approx(expected)
    assert result["excluded_components"] == []


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), None, 0.0, -5.0])
def test_one_broken_model_does_not_corrupt_the_ensemble(bad_value):
    """
    A NaN/inf/missing/non-positive prediction from one model used to make the
    whole blended price NaN. Now that component is excluded, its weight is
    redistributed, and the surviving model still produces a usable number.
    """
    result = ensemble.compute_ensemble_prediction(
        lr_price=105.0, lstm_price=bad_value, llm_panel_summary=None, current_price=100.0
    )
    assert np.isfinite(result["final_predicted_price"])
    assert result["final_predicted_price"] == pytest.approx(105.0)
    assert result["weights_used"]["lstm"] == 0.0
    assert result["weights_used"]["linear_regression"] == pytest.approx(1.0)
    # The failure is reported, never silently swallowed.
    assert [e["component"] for e in result["excluded_components"]] == ["lstm"]


def test_surviving_weights_always_sum_to_one():
    result = ensemble.compute_ensemble_prediction(
        lr_price=float("nan"), lstm_price=103.0,
        llm_panel_summary={"panel_price_estimate": 110.0}, current_price=100.0
    )
    total = sum(result["weights_used"][k] for k in ("linear_regression", "lstm", "llm_panel"))
    assert total == pytest.approx(1.0)
    assert result["weights_used"]["linear_regression"] == 0.0


def test_every_model_failing_raises_instead_of_returning_a_nan_price():
    """
    A NaN price would flow onward into signal_generator, the webapp and
    report.json looking like a real number. Fail at the point of the problem.
    """
    with pytest.raises(ensemble.AllModelsFailedError):
        ensemble.compute_ensemble_prediction(
            lr_price=float("nan"), lstm_price=None,
            llm_panel_summary=None, current_price=100.0
        )


def test_ensemble_is_deterministic_for_identical_inputs():
    kwargs = dict(lr_price=99.5, lstm_price=102.25,
                  llm_panel_summary={"panel_price_estimate": 101.0}, current_price=100.0)
    first = ensemble.compute_ensemble_prediction(**kwargs)
    for _ in range(5):
        assert ensemble.compute_ensemble_prediction(**kwargs) == first


def test_dynamic_weights_are_deterministic_for_identical_inputs():
    class _WF:
        def __init__(self, rmse, n):
            self.mean_rmse = rmse
            self.folds = list(range(n))

    lr, lstm = _WF(2.0, 5), _WF(4.0, 5)
    first = ensemble.compute_dynamic_weights(lr, lstm, min_folds_required=3)
    for _ in range(5):
        assert ensemble.compute_dynamic_weights(lr, lstm, min_folds_required=3) == first
    # Lower validation RMSE must earn the larger share.
    assert first["linear_regression"] > first["lstm"]
    assert sum(first[k] for k in ("linear_regression", "lstm", "llm_panel")) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. Export robustness
# ---------------------------------------------------------------------------
def test_csv_export_survives_a_missing_metric():
    """
    A missing metric key used to reach the :.4f format spec as an empty
    string and raise ValueError, turning it into a failed download.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "webapp"))
    import export_utils

    payload = {
        "ticker": "TESTX",
        "lr_eval": {"rmse": 1.0},  # deliberately missing mae/r2/mape/accuracy_pct
        "lstm_eval": {},           # missing everything
        "final_ensemble": {"current_price": 100.0, "final_predicted_price": 101.0,
                            "predicted_pct_change": 1.0},
        "final_signal": {"action": "HOLD", "reasoning": "test"},
        "export_rows": [{"date": "2024-01-02", "actual": 100.0,
                          "lr_predicted": 101.0, "lstm_predicted": None}],
    }
    csv_text = export_utils.build_csv(payload)
    assert "Directional Accuracy (%)" in csv_text
    assert "n/a" in csv_text
    assert "TESTX" in csv_text
