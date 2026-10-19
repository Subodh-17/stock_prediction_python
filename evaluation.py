"""
evaluation.py -- fair, same-window model comparison.

Everything here is computed on ONE set of out-of-sample target dates shared by every model, with the
same target (next-day close, in price AND return space) and the same adjustment convention, so the
numbers are directly comparable. Models compared:

  naive            predict tomorrow's close = today's close (the baseline every model must beat)
  drift            today's close * (1 + mean daily return of data strictly BEFORE the first test date)
  linear_regression, lstm, lr_lstm_ensemble   (as produced by the pipeline's chronological split)

A metric that cannot be defined is returned as None with a reason in `unavailable` -- never as 0.
Training-set performance is never reported here. `period` and `n_obs` always travel with the metrics.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config

MAX_GAP_DAYS = 5               # calendar days between a target bar and its base bar; a long weekend is 4, a holiday cluster 5
MIN_OBS_EVAL = 30              # below this the model comparison is reported as "Insufficient evaluation data"
MIN_OBS_METRICS = 10
MIN_OBS_INTERVAL = 80          # >= 40 calibration + >= 40 held-out for the coverage check
INTERVAL_LEVEL = 0.80


def _d(x) -> str:
    return str(x)[:10]


def _sign(a):
    return np.sign(a)


def directional_accuracy_vs_base(actual, pred, base):
    """Share of days where sign(pred-base)==sign(actual-base). Flat actual moves are excluded
    (direction undefined). Returns (pct or None, n_used, reason or None)."""
    da, dp = actual - base, pred - base
    mask = da != 0
    n = int(mask.sum())
    if n < MIN_OBS_METRICS:
        return None, n, f"fewer than {MIN_OBS_METRICS} non-flat days"
    if np.all(dp[mask] == 0):
        return None, n, "model makes no directional call (predicts no change)"
    return float(np.mean(_sign(da[mask]) == _sign(dp[mask])) * 100), n, None


def _mape(actual, pred):
    ok = np.abs(actual) > 1e-9
    if ok.sum() < MIN_OBS_METRICS:
        return None
    return float(np.mean(np.abs((actual[ok] - pred[ok]) / actual[ok])) * 100)


def _dm_vs_naive(err_model, err_naive):
    """Approximate Diebold-Mariano z for squared-error loss, h=1 (no HAC; large-sample normal).
    Positive z = model has LOWER loss than naive. Returned with its caveat; None if undefined."""
    d = err_naive ** 2 - err_model ** 2
    n = len(d)
    sd = d.std(ddof=1) if n > 2 else 0.0
    if n < 30 or not sd > 0:
        return None
    z = float(d.mean() / (sd / math.sqrt(n)))
    p = float(math.erfc(abs(z) / math.sqrt(2)))      # two-sided
    return {"z": round(z, 3), "p_two_sided": round(p, 4),
            "note": "approximate (no autocorrelation correction); treat as indicative only"}


BOOTSTRAP_B = 1000


def _block_len(n: int) -> int:
    """Moving-block length for the bootstrap: ~2*n^(1/3), at least 5. A heuristic for serially dependent data, not an optimum."""
    return max(5, int(round(2 * n ** (1 / 3))))


def _block_indices(n: int, B: int, rng) -> np.ndarray:
    L = _block_len(n)
    nb = -(-n // L)
    starts = rng.integers(0, max(1, n - L + 1), size=(B, nb))
    return (starts[:, :, None] + np.arange(L)).reshape(B, -1)[:, :n]


def bootstrap_rmse_improvement_cis(err_model: np.ndarray, err_naive: np.ndarray, levels=(0.95, 0.99), B: int = BOOTSTRAP_B,
                                   seed: int = None):
    """Percentile CIs (in %) for 1 - RMSE_model/RMSE_naive from ONE moving-block bootstrap over the paired out-of-sample errors
    (pairing keeps cross-model correlation; blocks keep short-range serial dependence). Deterministic (seeded).
    Returns {level: [lo, hi]} or None when n < MIN_OBS_EVAL or the naive RMSE is zero.

    CALIBRATION (measured in V20, synthetic experiment: iid-normal errors with log-AR(1) volatility, NO true skill, n=60-250,
    300 replications): the 95% interval wrongly reported "model beats naive" (lower bound > 0) in 4-8% of trials versus a nominal
    2.5% -- i.e. it is anti-conservative; the 99% interval did so in 0.3-3.3%. The verdict therefore uses the 99% interval."""
    n = len(err_model)
    if n < MIN_OBS_EVAL or not np.sum(err_naive ** 2) > 0:
        return None
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    idx = _block_indices(n, B, rng)
    rm = np.sqrt(np.mean(err_model[idx] ** 2, axis=1))
    rn = np.sqrt(np.mean(err_naive[idx] ** 2, axis=1))
    ok = rn > 0
    imp = (1 - rm[ok] / rn[ok]) * 100
    out = {}
    for lv in levels:
        a = (1 - lv) / 2 * 100
        out[lv] = [float(np.percentile(imp, a)), float(np.percentile(imp, 100 - a))]
    return out


def bootstrap_rmse_improvement_ci(err_model, err_naive, level: float = 0.95, B: int = BOOTSTRAP_B, seed: int = None):
    r = bootstrap_rmse_improvement_cis(err_model, err_naive, levels=(level,), B=B, seed=seed)
    return None if r is None else r[level]


def bootstrap_directional_ci(hits: np.ndarray, level: float = 0.95, B: int = BOOTSTRAP_B, seed: int = None):
    """Block-bootstrap CI (in %) for directional accuracy over the days where a direction is defined (`hits` are 0/1)."""
    n = len(hits)
    if n < MIN_OBS_EVAL:
        return None
    rng = np.random.default_rng(config.RANDOM_SEED if seed is None else seed)
    acc = hits[_block_indices(n, B, rng)].mean(axis=1) * 100
    a = (1 - level) / 2 * 100
    return [float(np.percentile(acc, a)), float(np.percentile(acc, 100 - a))]


def _skew(x):
    s = x.std()
    return float(np.mean(((x - x.mean()) / s) ** 3)) if s > 0 and len(x) > 2 else None


def _lag1(x):
    if len(x) < 10 or x.std() == 0:
        return None
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def aligned_frame(result: dict) -> pd.DataFrame:
    """Date-indexed frame: actual, base (previous close), and each model's predicted close."""
    df: pd.DataFrame = result["df"]
    if getattr(df.index, "tz", None) is not None:       # defence in depth: compare naive exchange-local dates only
        df = df.copy(); df.index = df.index.tz_localize(None)
    close = df["Close"]
    lr = pd.Series(np.asarray(result["lr_eval"]["predictions"], float),
                   index=[_d(x) for x in result["lr_dates_test"]])
    act = pd.Series(np.asarray(result["y_test"], float), index=[_d(x) for x in result["lr_dates_test"]])
    ls_eval = result["lstm_eval"]
    ls = pd.Series(np.asarray(ls_eval["predictions"], float), index=[_d(x) for x in result["lstm_dates_test"]])
    ls_act = pd.Series(np.asarray(ls_eval["actual"], float), index=[_d(x) for x in result["lstm_dates_test"]])
    common = sorted(set(lr.index) & set(ls.index))
    if len(common) < 2:
        raise ValueError("LR and LSTM test windows do not overlap enough to compare.")
    idx = {_d(d): i for i, d in enumerate(df.index)}
    rows, n_gap = [], 0
    for d in common:
        i = idx.get(d)
        if i is None or i == 0:
            continue
        gap = (df.index[i] - df.index[i - 1]).days
        if gap > MAX_GAP_DAYS:                      # missing bars: this is NOT a next-day target; exclude and count
            n_gap += 1
            continue
        rows.append({"date": d, "actual": float(close.iloc[i]), "base": float(close.iloc[i - 1]),
                     "linear_regression": lr[d], "lstm": ls[d], "_act_check": (act[d], ls_act[d])})
    f = pd.DataFrame(rows).set_index("date")
    # Guard: the three "actual" series must agree with the raw close (catches date-misalignment bugs).
    bad = np.abs(f["actual"] - [a[0] for a in f["_act_check"]]) > 1e-6 * f["actual"].abs().clip(lower=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} test dates are misaligned between the model outputs and raw closes.")
    f = f.drop(columns="_act_check")
    finite = np.isfinite(f[["actual", "base", "linear_regression", "lstm"]].to_numpy(float)).all(axis=1) & (f["base"].to_numpy(float) > 0)
    n_bad = int((~finite).sum())
    f = f.loc[finite]                                  # explicit, counted -- never silently imputed or zero-filled
    if len(f) < 2:
        raise ValueError("Fewer than 2 finite aligned out-of-sample observations.")
    w = config.ENSEMBLE_WEIGHTS
    tot = w["linear_regression"] + w["lstm"]
    wl = w["linear_regression"] / tot if tot > 0 else 0.5
    f["lr_lstm_ensemble"] = wl * f["linear_regression"] + (1 - wl) * f["lstm"]
    f["naive"] = f["base"]
    first = pd.Timestamp(f.index[0])
    past = close[close.index < first].pct_change().dropna()          # strictly before the test window
    mu = float(past.mean()) if len(past) >= 30 else None
    if mu is not None:
        f["drift"] = f["base"] * (1 + mu)
    train_diffs = close[close.index < first].diff().abs().dropna()
    f.attrs["naive_insample_mae"] = float(train_diffs.mean()) if len(train_diffs) >= 30 else None
    f.attrs["naive_insample_n"] = int(len(train_diffs))
    f.attrs["drift_mean_return"] = mu
    f.attrs["n_dropped_nonfinite"] = n_bad
    f.attrs["n_dropped_gap"] = n_gap
    return f


MODELS = ["naive", "drift", "linear_regression", "lstm", "lr_lstm_ensemble"]


def evaluate_frame(f: pd.DataFrame) -> dict:
    a, b = f["actual"].to_numpy(), f["base"].to_numpy()
    n = len(f)
    out: List[dict] = []
    naive_err = a - f["naive"].to_numpy()
    naive_rmse = float(np.sqrt(np.mean(naive_err ** 2)))
    for m in MODELS:
        if m not in f:
            out.append({"model": m, "n_obs": 0, "unavailable": {"all": "drift needs >=30 prior returns before the test window"}})
            continue
        p = f[m].to_numpy()
        err = a - p
        unav: Dict[str, str] = {}
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ret_a, ret_p = a / b - 1, p / b - 1
        dacc, dn, why = directional_accuracy_vs_base(a, p, b)
        if dacc is None:
            unav["directional_accuracy_pct"] = why
        mape = _mape(a, p)
        if mape is None:
            unav["mape"] = "undefined (near-zero actuals or too few observations)"
        imp = None
        if m != "naive":
            imp = float((1 - rmse / naive_rmse) * 100) if naive_rmse > 0 else None
            if imp is None:
                unav["rmse_improvement_vs_naive_pct"] = "naive RMSE is zero"
        dm = _dm_vs_naive(err, naive_err) if m != "naive" else None
        if m != "naive" and dm is None:
            unav["diebold_mariano"] = "needs >=30 observations and non-zero loss variance"
        res = err
        mase = None
        scale = f.attrs.get("naive_insample_mae")
        if scale and scale > 0:
            mase = float(np.mean(np.abs(err)) / scale)
        else:
            unav["mase"] = "needs >= 30 training-period price changes before the test window"
        cis = bootstrap_rmse_improvement_cis(err, naive_err) if m != "naive" else None
        imp_ci = cis[0.95] if cis else None
        imp_ci99 = cis[0.99] if cis else None
        if m != "naive" and imp_ci is None:
            unav["rmse_improvement_ci95"] = f"needs >= {MIN_OBS_EVAL} observations and a non-zero naive error"
        dmask = (a - b) != 0
        dir_ci = None
        if dacc is not None:
            hits = (np.sign((a - b)[dmask]) == np.sign((p - b)[dmask])).astype(float)
            dir_ci = bootstrap_directional_ci(hits)
        out.append({
            "model": m, "n_obs": n, "mae": float(np.mean(np.abs(err))), "rmse": rmse, "mape": mape,
            "return_mae": float(np.mean(np.abs(ret_a - ret_p))), "return_rmse": float(np.sqrt(np.mean((ret_a - ret_p) ** 2))),
            "directional_accuracy_pct": dacc, "directional_n": dn,
            "rmse_improvement_vs_naive_pct": imp, "rmse_improvement_ci95": imp_ci, "rmse_improvement_ci99": imp_ci99, "directional_accuracy_ci95": dir_ci,
            "mase": mase, "diebold_mariano_vs_naive": dm,
            "bias": float(np.mean(-err)),
            "residuals": {"mean": float(res.mean()), "std": float(res.std(ddof=1)) if n > 1 else None,
                          "skew": _skew(res), "lag1_autocorr": _lag1(res)},
            "unavailable": unav,
        })
    return {"horizon": {"days": 1, "definition": "forecast of the NEXT trading day's close, made at the close of the previous trading day; "
                        "one horizon only -- the models do not produce multi-day forecasts, so none are mixed in"},
            "mase_note": ("MASE = MAE / mean |daily change| of the training period before the test window (Hyndman & Koehler). "
                          "Below 1 = better than the in-sample naive forecast's typical error."),
            "ci_note": (f"Moving-block bootstrap intervals (B={BOOTSTRAP_B}, block ~2*n^(1/3), seeded) over paired out-of-sample errors. "
                        f"The verdict uses the 99% interval because the 95% one proved anti-conservative in a calibration experiment "
                        f"(see docs); all intervals are indicative and assume only short-range dependence."),
            "models": out, "n_obs": n, "period": {"start": f.index[0], "end": f.index[-1]},
            "target": "next-trading-day close (price) and close-to-close return",
            "drift_mean_return_used": f.attrs.get("drift_mean_return"),
            "out_of_sample": True}


# ---------------------------------------------------------------------------------------------------------------
# Prediction intervals: rolling, strictly-past, chronological conformal calibration
# ---------------------------------------------------------------------------------------------------------------
# Roles of the data (all chronological; nothing is shuffled):
#   TRAIN        rows the LR/LSTM were fitted on (pipeline's train split). Never used for calibration or scoring.
#   RESIDUAL     one absolute return residual |actual/base - predicted/base| per OUT-OF-SAMPLE target date. Each comes
#                from a model that never saw that day (models are fitted once, on TRAIN only).
#   CALIBRATION  for the interval of target date t: the most recent `window` residuals with date STRICTLY BEFORE t
#                (the residual of day t-1 is known once day t-1 closes, i.e. before the day-t forecast is made).
#   EVALUATION   target dates t with at least `min_cal` earlier residuals. Their coverage is measured against the
#                SAME intervals that were produced -- the interval shown for a historical day IS the interval scored.
# The interval for a NEW forecast is produced by the identical procedure from the latest `window` residuals (all of
# which precede the forecast date). Its own coverage is unknowable until it is realised; what is reported next to it is
# the historical coverage of the PROCEDURE, which is evidence about the method, not a guarantee for this interval.
#
# Limitation, stated plainly: split-conformal coverage guarantees assume exchangeable residuals. Financial returns are
# serially dependent (volatility clustering) and non-stationary, so NO finite-sample coverage guarantee is claimed.
# A rolling window adapts to slow regime changes; it cannot anticipate sudden ones. The measured gap is shown.
CONFORMAL_WINDOW = 120
CONFORMAL_MIN_CAL = 40
CONFORMAL_MIN_EVAL = 30


def _abs_return_residuals(f: pd.DataFrame, model: str):
    """Chronological (dates, |return residual|, base, point, actual) with non-finite rows REMOVED and counted."""
    if model not in f:
        return None
    base = f["base"].to_numpy(float); act = f["actual"].to_numpy(float); pred = f[model].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.abs(act / base - pred / base)
    ok = np.isfinite(res) & np.isfinite(base) & (base > 0) & np.isfinite(pred) & np.isfinite(act)
    dates = np.array([str(d) for d in f.index])
    return dates[ok], res[ok], base[ok], pred[ok], act[ok], int((~ok).sum())


def _conformal_q(past: np.ndarray, level: float):
    """Finite-sample-corrected empirical quantile (rank ceil((m+1)*level)). Returns None when the rank exceeds m,
    i.e. when `m` residuals cannot support that level -- never silently clamped to the maximum."""
    m = len(past)
    k = math.ceil((m + 1) * level)
    if m == 0 or k > m:
        return None
    return float(np.partition(past, k - 1)[k - 1])


def rolling_conformal(f: pd.DataFrame, model: str, level: float = INTERVAL_LEVEL, window: int = CONFORMAL_WINDOW,
                      min_cal: int = CONFORMAL_MIN_CAL, min_eval: int = CONFORMAL_MIN_EVAL) -> dict:
    """Intervals for every evaluable out-of-sample date, built ONLY from earlier residuals, and their measured
    coverage/width. Returns status 'insufficient_evaluation_data' (with the counts) when it cannot be done honestly."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    r = _abs_return_residuals(f, model)
    if r is None:
        return {"status": "unavailable", "message": f"model '{model}' has no out-of-sample predictions"}
    dates, res, base, pred, act, dropped = r
    n = len(res)
    series, covered, widths, unavailable = [], 0, [], 0
    for i in range(min_cal, n):
        past = res[max(0, i - window):i]                       # strictly before i; at most `window` most recent
        q = _conformal_q(past, level)
        if q is None:
            unavailable += 1
            continue
        lo, hi = pred[i] - base[i] * q, pred[i] + base[i] * q
        hit = bool(lo <= act[i] <= hi)
        covered += hit
        widths.append(2 * q)
        series.append({"date": dates[i], "low": float(lo), "high": float(hi), "point": float(pred[i]),
                       "actual": float(act[i]), "covered": hit, "q_return": q, "n_calibration": int(len(past))})
    ne = len(series)
    base_info = {"model": model, "nominal_coverage": level, "window": window, "min_calibration": min_cal,
                 "n_residuals": n, "n_nonfinite_dropped": dropped, "n_unavailable_level_unsupported": unavailable,
                 "method": ("rolling split-conformal on absolute return residuals; each interval uses only the most "
                            f"recent {window} residuals dated strictly before its target date; "
                            "finite-sample rank ceil((m+1)*level)"),
                 "caveat": ("No coverage guarantee is claimed: returns are serially dependent and non-stationary, "
                            "which violates the exchangeability assumption behind conformal guarantees.")}
    if ne < min_eval:
        return {**base_info, "status": "insufficient_evaluation_data",
                "message": (f"Insufficient evaluation data: {n} usable out-of-sample residuals give {ne} evaluable "
                            f"intervals; at least {min_eval} are required after the first {min_cal} calibration residuals."),
                "n_evaluated": ne}
    obs = covered / ne
    return {**base_info, "status": "ok", "n_evaluated": ne, "n_covered": int(covered),
            "evaluation_period": {"start": series[0]["date"], "end": series[-1]["date"]},
            "observed_coverage": obs, "coverage_gap": obs - level,
            "mean_relative_width": float(np.mean(widths)),
            "mean_width_price": float(np.mean([s["high"] - s["low"] for s in series])),
            "assessment": ("under-covering (intervals were too narrow)" if obs < level - 0.10 else
                           "over-covering (intervals were wider than needed)" if obs > level + 0.10 else
                           "close to nominal over this period"),
            "series": series}


def forecast_interval(f: pd.DataFrame, model: str, point_price: float, base_price: float,
                      level: float = INTERVAL_LEVEL, window: int = CONFORMAL_WINDOW,
                      min_cal: int = CONFORMAL_MIN_CAL, min_eval: int = CONFORMAL_MIN_EVAL) -> dict:
    """Interval for the NEXT, not-yet-observed target. Calibrated on the latest `window` residuals, all of which are
    dated before that target. Reported with the historical performance of the same procedure -- and marked
    unavailable if that evidence is insufficient."""
    evaluation = rolling_conformal(f, model, level, window, min_cal, min_eval)
    if evaluation["status"] != "ok":
        return {"status": evaluation["status"], "level": level, "message": evaluation.get("message"),
                "nominal_coverage": level, "evaluation": {k: v for k, v in evaluation.items() if k != "series"}}
    dates, res, *_ = _abs_return_residuals(f, model)
    past = res[-window:]
    q = _conformal_q(past, level)
    if q is None:
        return {"status": "insufficient_evaluation_data", "level": level, "nominal_coverage": level,
                "message": f"{len(past)} residuals cannot support a {level:.0%} interval.",
                "evaluation": {k: v for k, v in evaluation.items() if k != "series"}}
    ev = {k: v for k, v in evaluation.items() if k != "series"}
    return {"status": "ok", "level": level, "nominal_coverage": level, "model": model,
            "low": float(point_price - base_price * q), "high": float(point_price + base_price * q), "q_return": q,
            "calibration": {"n": int(len(past)), "start": str(dates[-len(past):][0]), "end": str(dates[-1]),
                            "note": "all calibration residuals are dated before the forecast target"},
            "evaluation": ev,
            "reading_guide": ("'evaluation' is the historical coverage of this PROCEDURE over the dates shown. It "
                              "does not validate this particular interval; its realised coverage is unknown until the "
                              "target date has passed."),
            "caveat": evaluation["caveat"]}


def conformal_interval(f: pd.DataFrame, model: str, point_price: float, current_price: float,
                       level: float = INTERVAL_LEVEL, min_obs: int = None) -> dict:
    """Backward-compatible name for forecast_interval (V17 callers). `min_obs` is ignored: the sufficiency rules
    are min_cal/min_eval, stated in the result."""
    return forecast_interval(f, model, point_price, current_price, level)


def build_evaluation(result: dict) -> dict:
    f = aligned_frame(result)
    ev = evaluate_frame(f)
    rows = {r["model"]: r for r in ev["models"]}
    ens = rows.get("lr_lstm_ensemble", {})
    ev["residual_series"] = {m: [{"date": d, "residual": float(v)} for d, v in (f["actual"] - f[m]).items()]
                             for m in ("lr_lstm_ensemble", "naive") if m in f}
    ev["series"] = [{"date": d, "actual": float(r.actual), **{m: float(r[m]) for m in MODELS if m in f}}
                    for d, r in f.iterrows()]
    ev["interval_evaluation"] = rolling_conformal(f, "lr_lstm_ensemble")
    ev["walk_forward"] = {"linear_regression": result.get("walk_forward_lr"), "lstm": result.get("walk_forward_lstm")}
    ev["n_dropped_nonfinite"] = f.attrs.get("n_dropped_nonfinite", 0)
    ev["n_dropped_gap"] = f.attrs.get("n_dropped_gap", 0)
    ev["max_gap_days"] = MAX_GAP_DAYS
    ev["sufficient"] = ev["n_obs"] >= MIN_OBS_EVAL
    ev["status"] = "ok" if ev["sufficient"] else "insufficient_evaluation_data"
    ev["conclusion"] = _conclusion(rows) if ev["sufficient"] else (
        f"Insufficient evaluation data: {ev['n_obs']} out-of-sample days (need >= {MIN_OBS_EVAL}); no verdict is given.")
    ev["_frame"] = f
    return ev


def _conclusion(rows: dict) -> str:
    """'Beats naive' requires the 99% interval's lower bound to be above zero (see bootstrap_rmse_improvement_cis)."""
    parts = []
    for m in ("linear_regression", "lstm", "lr_lstm_ensemble"):
        r = rows.get(m, {})
        imp, c95, c99 = r.get("rmse_improvement_vs_naive_pct"), r.get("rmse_improvement_ci95"), r.get("rmse_improvement_ci99")
        if imp is None:
            continue
        span = (f"; 95% CI [{c95[0]:+.1f}%, {c95[1]:+.1f}%], 99% CI [{c99[0]:+.1f}%, {c99[1]:+.1f}%]" if c95 and c99
                else "; no confidence interval (too few observations)")
        if c99 and c99[0] > 0:
            parts.append(f"{m}: beats the naive last-close baseline on RMSE ({imp:+.1f}%{span})")
        elif c99 and c99[1] < 0:
            parts.append(f"{m}: does NOT beat the naive last-close baseline on RMSE -- it is WORSE ({imp:+.1f}%{span})")
        elif c99:
            parts.append(f"{m}: does NOT beat the naive last-close baseline on RMSE -- not distinguishable from it ({imp:+.1f}%{span})")
        else:
            parts.append(f"{m}: does NOT beat the naive last-close baseline on RMSE (point estimate {imp:+.1f}%{span})" if imp <= 0
                         else f"{m}: {imp:+.1f}% vs naive on RMSE, but with no confidence interval this is not evidence of skill")
    return "; ".join(parts) or "No model comparison could be computed."
