"""
forecast_service.py -- runs the (slow) forecast pipeline OFF the request thread.

  * Jobs run in a single background worker; the web request returns a job id immediately.
  * Identical requests share one job; a finished forecast is cached by (instrument, period, epochs,
    LAST COMPLETED BAR, model config). A quote refresh never touches this cache, so it never retrains.
  * The LSTM is reused from an auto-named checkpoint when compatible (pipeline enforces compatibility).
  * A bar for the current, still-open session is dropped before modelling (it is not a settled close).
  * Cancellation is cooperative: checked between stages (a running Keras fit is not interrupted).
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import numpy as np

import config
import market_data
import perf
import security

_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forecast")
_lock = threading.RLock()     # re-entrant: check-and-register must be ONE atomic section (see submit_forecast)
JOBS: "OrderedDict[str, dict]" = OrderedDict()
_RESULTS: "OrderedDict[str, dict]" = OrderedDict()      # cache_key -> {"payload", "frame", "df", "at"}
_ACTIVE_BY_KEY: Dict[str, str] = {}
_MAX_JOBS, _MAX_RESULTS = 60, 8


class Cancelled(Exception):
    pass


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)[:40]


from market_session import drop_partial_bar   # noqa: E402,F401  (re-exported: V17/V18 callers use forecast_service.drop_partial_bar)


def _new_job(kind, key) -> dict:
    job = {"id": uuid.uuid4().hex[:24], "kind": kind, "key": key, "status": "queued", "stage": "queued",
           "created": time.time(), "started": None, "finished": None, "error": None, "cancel": False,
           "result_key": None, "timings": {}, "_event": threading.Event()}
    with _lock:
        JOBS[job["id"]] = job
        while len(JOBS) > _MAX_JOBS:
            JOBS.popitem(last=False)
    return job


def public_job(job: dict) -> dict:
    out = {k: v for k, v in job.items() if k != "cancel" and not k.startswith("_")} | {"cancel_requested": job["cancel"]}
    if job.get("started"):
        out["elapsed_seconds"] = round((job.get("finished") or time.time()) - job["started"], 1)
    # A running job past the limit is REPORTED as timed out so a UI never polls forever. The worker thread cannot be
    # killed mid-fit; if it later finishes, its result is simply never claimed by this request flow.
    if job["status"] == "running" and time.time() - job["started"] > config.JOB_TIMEOUT_SECONDS:
        out["status"], out["timed_out"] = "timeout", True
    return out


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        return JOBS.get(job_id)


import logging_utils
_log = logging_utils.get_logger("forecast_service")    # namespaced: the project's redaction filter applies, incl. tracebacks


def _safe_error_text(e: BaseException) -> str:
    """User-visible failure text. Only exception types whose messages are written to be shown to users are passed
    through (still secret-redacted and bounded); anything else becomes a generic message, with the traceback going
    to the server log only (it could contain file paths or fragments of upstream responses)."""
    safe = [ValueError, LookupError, market_data.ProviderError, market_data.RateLimited, market_data.InvalidSymbol]
    for modname, clsname in (("data_loader", "InsufficientDataError"), ("data_sources.base", "DataSourceError"),
                             ("ensemble", "AllModelsFailedError")):
        try:
            safe.append(getattr(__import__(modname, fromlist=[clsname]), clsname))
        except (ImportError, AttributeError):
            pass
    if isinstance(e, tuple(safe)):
        return security.redact(f"{e}")[:400]
    return "An unexpected error occurred. Details are in the server log."


class TaskError(Exception):
    """A failure whose message is written to be shown to the user as-is (already safe, no internals)."""
    def __init__(self, message: str, http_status: int = 500, code: str = None):
        super().__init__(message)
        self.user_message, self.http_status, self.code = message, http_status, code


def submit_task(kind: str, key: str, fn: Callable) -> dict:
    """Run `fn(job)` on the background worker. Identical in-flight `key`s share one job (no duplicates). The return
    value of fn becomes job['result_key']. A failed job NEVER exposes a previous result."""
    with _lock:
        running = _ACTIVE_BY_KEY.get(key)
        if running and JOBS.get(running, {}).get("status") in ("queued", "running"):
            return JOBS[running]
        job = _new_job(kind, key)
        _ACTIVE_BY_KEY[key] = job["id"]
    _EXEC.submit(_task_worker, job, key, fn)
    return job


def _task_worker(job, key, fn):
    try:
        if job["cancel"]:
            return
        job["status"], job["started"], job["stage"] = "running", time.time(), "running"
        job["result_key"] = fn(job)
        job["status"], job["stage"] = "done", "done"
    except Cancelled:
        job["status"], job["stage"] = "cancelled", "cancelled"
    except TaskError as e:
        job["status"], job["stage"], job["error"], job["http_status"] = "failed", "failed", security.redact(e.user_message)[:400], e.http_status
        job["error_code"] = e.code
    except Exception as e:  # noqa: BLE001
        job["status"], job["stage"], job["http_status"] = "failed", "failed", 500
        job["error"] = _safe_error_text(e)
        _log.error("task %s failed: %s", job["id"], security.redact(f"{type(e).__name__}: {e}"), exc_info=True)
    finally:
        job["finished"] = job.get("finished") or time.time()
        with _lock:
            if _ACTIVE_BY_KEY.get(key) == job["id"]:
                _ACTIVE_BY_KEY.pop(key, None)
        job["_event"].set()


def wait_for(job: dict, seconds: float) -> bool:
    """True if the job reached a terminal state within `seconds` (bounded; never waits indefinitely)."""
    ev = job.get("_event")
    if ev is not None:
        ev.wait(max(0.0, seconds))
    return job["status"] in ("done", "failed", "cancelled")


def cancel_job(job_id: str) -> bool:
    job = get_job(job_id)
    if not job or job["status"] in ("done", "failed", "cancelled"):
        return False
    job["cancel"] = True
    if job["status"] == "queued":
        job["status"], job["stage"], job["finished"] = "cancelled", "cancelled before start", time.time()
        if job.get("_event"):
            job["_event"].set()
    return True


def cache_key(inst, period, epochs, last_bar, walk_forward=False) -> str:
    return "|".join([inst.key, period, str(epochs), last_bar, "wf" if walk_forward else "nowf", str(config.LOOKBACK_WINDOW),
                     config.LSTM_PREDICTION_TARGET, str(config.TRAIN_TEST_SPLIT)])


def submit_forecast(raw_symbol: str, period: str = "2y", epochs: Optional[int] = None,
                    svc: Optional[market_data.MarketDataService] = None, run: Optional[Callable] = None,
                    walk_forward: bool = False) -> dict:
    """Returns a job dict. Cheap checks happen here; the heavy work is queued."""
    svc = svc or market_data.get_service()
    inst = market_data.resolve_instrument(raw_symbol)
    epochs = int(epochs or config.LSTM_EPOCHS)
    if not 1 <= epochs <= 200:
        raise ValueError("epochs must be between 1 and 200")
    hist = svc.get_history(raw_symbol, period)           # cached; also tells us the last bar (cache key)
    last_bar = hist["last_bar"]
    key = cache_key(inst, period, epochs, last_bar, walk_forward)
    with _lock:      # cache lookup, in-flight lookup and registration are ONE atomic section: no duplicate jobs
        hit = _RESULTS.get(key)
        if hit and time.time() - hit["at"] < config.FORECAST_CACHE_TTL_SECONDS:
            job = {"id": uuid.uuid4().hex[:24], "kind": "forecast", "key": key, "status": "done", "stage": "cache hit",
                   "created": time.time(), "started": time.time(), "finished": time.time(), "error": None,
                   "cancel": False, "result_key": key, "timings": {"cached": True}}
            JOBS[job["id"]] = job
            return job
        running = _ACTIVE_BY_KEY.get(key)
        if running and JOBS.get(running, {}).get("status") in ("queued", "running"):
            return JOBS[running]
        job = _new_job("forecast", key)
        _ACTIVE_BY_KEY[key] = job["id"]
    _EXEC.submit(_worker, job, raw_symbol, inst, period, epochs, key, svc, run or _run_pipeline_and_evaluate, walk_forward)
    return job


def _worker(job, raw_symbol, inst, period, epochs, key, svc, run, walk_forward=False):
    if job["cancel"]:
        return
    job["status"], job["started"] = "running", time.time()
    try:
        _t_job = time.perf_counter()
        payload, stash = run(job, raw_symbol, inst, period, epochs, svc, walk_forward)
        perf.record("forecast.job_total", (time.perf_counter() - _t_job) * 1000)
        with _lock:
            _RESULTS[key] = {"payload": payload, "at": time.time(), **stash}
            while len(_RESULTS) > _MAX_RESULTS:
                _RESULTS.popitem(last=False)
        job["result_key"], job["status"], job["stage"] = key, "done", "done"
    except Cancelled:
        job["status"], job["stage"] = "cancelled", "cancelled"
    except Exception as e:  # noqa: BLE001
        job["status"], job["stage"] = "failed", "failed"
        job["error"] = _safe_error_text(e)
        _log.error("forecast job %s failed: %s", job["id"], security.redact(f"{type(e).__name__}: {e}"), exc_info=True)
    finally:
        job["finished"] = time.time()
        with _lock:
            if _ACTIVE_BY_KEY.get(key) == job["id"]:
                _ACTIVE_BY_KEY.pop(key, None)
        if job.get("_event"):
            job["_event"].set()


def _stage(job, name):
    if job["cancel"]:
        raise Cancelled()
    job["stage"] = name
    job["_t"] = time.perf_counter()


def _end(job, name):
    job["timings"][name] = round((time.perf_counter() - job.get("_t", time.perf_counter())) * 1000, 1)


def _run_pipeline_and_evaluate(job, raw_symbol, inst, period, epochs, svc, walk_forward=False):
    import checkpoints as ck
    import evaluation
    import pipeline
    import data_loader

    _stage(job, "loading history")
    hist = svc.get_history(raw_symbol, period)
    import pandas as pd
    df = pd.DataFrame(hist["bars"]).rename(columns={"t": "Date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    q = svc.get_quote(raw_symbol)
    df, dropped = drop_partial_bar(df, inst, q.session)
    data_loader.validate_sufficient_data(df)
    _end(job, "history")

    _stage(job, "training / loading models")
    ck_name = f"auto_{_safe_name(inst.key)}_{_safe_name(period)}_{config.LSTM_PREDICTION_TARGET}_{config.LOOKBACK_WINDOW}"
    result = pipeline.run_pipeline(ticker=inst.yahoo_symbol, period=period, epochs=epochs, skip_llm=True, verbose=False,
                                   checkpoint_name=ck_name, df=df, walk_forward=walk_forward)
    _end(job, "models")

    _stage(job, "evaluating")
    ev = evaluation.build_evaluation(result)
    frame = ev.pop("_frame")
    fe = result["final_ensemble"]
    cur = float(result["current_price"])
    interval = evaluation.conformal_interval(frame, "lr_lstm_ensemble", fe["final_predicted_price"], cur)
    _end(job, "evaluation")

    meta = result["run_metadata"]
    payload = {
        "instrument": inst.to_dict(), "generated_ts": datetime.now(timezone.utc).isoformat(),
        "as_of_bar": str(df.index[-1])[:10], "last_close": cur, "dropped_partial_bar": dropped,
        "horizon": "next trading day after the as-of bar",
        "target": {"lr": "next-day close (price level)", "lstm": f"next-day {config.LSTM_PREDICTION_TARGET}",
                   "evaluation": ev["target"], "adjustment": "split/dividend adjusted closes",
                   "currency": q.currency},
        "components": fe["component_predictions"], "weights_used": fe["weights_used"],
        "final_price": fe["final_predicted_price"], "pct_change": fe["predicted_pct_change"],
        "excluded_components": fe["excluded_components"],
        "interval": interval,
        "heuristic_confidence_legacy": {"label": result["confidence"]["confidence_label"],
                                        "note": "Legacy HEURISTIC from model agreement + RMSE. Not a probability; do not read it as accuracy."},
        "model_agreement": result["confidence"]["model_agreement"],
        "training": {"lstm_source": "reused checkpoint" if result["lstm_checkpoint"]["reused"] else "trained this run",
                     "checkpoint": ck_name, "lstm_training_data_ends": result["lstm_checkpoint"]["train_end"],
                     "first_evaluated_date": result["lstm_checkpoint"]["first_eval_date"],
                     "train_rows": (meta.get("data_shape") or {}).get("n_train_rows"),
                     "test_rows": (meta.get("data_shape") or {}).get("n_test_rows"),
                     "lookback": (meta.get("model_config") or {}).get("lookback_window"),
                     "train_test_split": (meta.get("model_config") or {}).get("train_test_split"),
                     "seed": meta.get("random_seed"),
                     "epochs": None if result["lstm_checkpoint"]["reused"] else epochs,
                     "data_first_bar": str(df.index[0])[:10], "data_last_bar": str(df.index[-1])[:10],
                     "run_metadata": meta},
        "data_quality": result["data_quality"], "data_provider": hist["provider"],
        "evaluation_summary": ev["conclusion"], "timings_ms": dict(job["timings"]),
        "disclaimer": "Statistical estimate from historical prices. Not advice; no claim of profitability or accuracy.",
        "evaluation": ev,
    }
    return payload, {"frame": frame, "df": df}


def get_result(key: str) -> Optional[dict]:
    with _lock:
        return _RESULTS.get(key)


# ---------------------------------------------------------------------------
# Backtest on the SAME aligned out-of-sample frame the evaluation used
# ---------------------------------------------------------------------------
BACKTEST_MODELS = ("linear_regression", "lstm", "lr_lstm_ensemble")


def run_backtest_job(key: str, params: dict) -> dict:
    import backtest as bt
    stash = get_result(key)
    if not stash:
        raise LookupError("No forecast result for this key. Run a forecast first.")
    model = params.get("model", "lr_lstm_ensemble")
    if model not in BACKTEST_MODELS:
        raise ValueError(f"model must be one of {BACKTEST_MODELS}")
    f, df = stash["frame"], stash["df"]
    start, end = params.get("start"), params.get("end")
    sub = f
    if start:
        sub = sub[sub.index >= start]
    if end:
        sub = sub[sub.index <= end]
    if len(sub) < 5:
        raise ValueError("Fewer than 5 out-of-sample days in the selected range.")
    # The backtest's day i uses base[i] = close[i-1]: prepend the day before the first test date so day 0 is a true close.
    first_i = df.index.get_loc(__import__("pandas").Timestamp(sub.index[0]))
    pre_date = str(df.index[first_i - 1])[:10]
    pre_close = float(df["Close"].iloc[first_i - 1])
    dates = [pre_date] + list(sub.index)
    actual = [pre_close] + list(sub["actual"])
    pred = [pre_close] + list(sub[model])
    opens = [float("nan")] + [float(df["Open"].loc[__import__("pandas").Timestamp(d)]) for d in sub.index]
    _t_bt = time.perf_counter()
    res = bt.run_backtest(
        dates, actual, pred, strategy_name=model,
        initial_capital=float(params.get("initial_capital", config.BACKTEST_INITIAL_CAPITAL)),
        transaction_cost_pct=float(params.get("transaction_cost_pct", config.BACKTEST_TRANSACTION_COST_PCT)),
        buy_threshold_pct=float(params.get("buy_threshold_pct", config.BUY_THRESHOLD_PCT)),
        sell_threshold_pct=float(params.get("sell_threshold_pct", config.SELL_THRESHOLD_PCT)),
        position_size_pct=float(params.get("position_size_pct", 1.0)),
        execution=params.get("execution", "next_open"), open_prices=opens,
        slippage_bps=float(params.get("slippage_bps", 2.0)))
    perf.record("backtest.run", (time.perf_counter() - _t_bt) * 1000)
    d = res.to_dict()
    d["window"] = {"start": dates[1], "end": dates[-1], "n_days": len(dates) - 1,
                   "note": "Held-out (out-of-sample) period of the models; the models never trained on these days."}
    d["caveats"] = ["Simulated execution; not a forecast of live results.",
                    "Adjusted prices; dividends are embedded in the adjusted series, taxes are not modelled.",
                    "Single instrument, long-only, whole-position entries/exits.",
                    "Strategy thresholds are user inputs; results from tuning them on this same window are in-sample."]
    return d


def warm_up() -> dict:
    """Pay the one-off cold-start cost (TensorFlow import + LSTM graph tracing) in the background at server start instead of
    inside the first user's forecast. Uses a tiny throw-away model on random numbers; nothing from it is ever stored or shown.
    Measured effect: see docs/V19_FINAL_AUDIT.md. Failure is harmless (the first forecast simply pays the cost)."""
    t0 = time.perf_counter()
    try:
        import numpy as np
        import checkpoints, data_loader, evaluation, pipeline  # noqa: F401  (imports ARE the warm-up: pulls in TensorFlow)
        from models.lstm_model import LSTMPredictor
        m = LSTMPredictor(lookback=5, units=4)
        rng = np.random.default_rng(0)
        X, y = rng.random((24, 5, 1)), rng.random(24)
        m.train(X, y, epochs=1, verbose=0)
        m.model.predict(X[:2], verbose=0)
        ok = True
    except Exception as e:  # noqa: BLE001
        _log.warning("warm-up failed (harmless): %s", security.redact(f"{type(e).__name__}: {e}"))
        ok = False
    ms = (time.perf_counter() - t0) * 1000
    perf.record("warmup.total", ms)
    return {"ok": ok, "ms": round(ms, 1)}
