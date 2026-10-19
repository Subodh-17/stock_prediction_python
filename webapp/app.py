"""
webapp/app.py
--------------
A Flask website for the Multi-AI Stock Prediction Ensemble -- a browser
front end for exactly the same pipeline.py / trading modules the CLI
(main.py, trade.py) already uses. Nothing about the underlying prediction
or trading logic is duplicated or reimplemented here; this file only adds
routes, templates, and request/response handling on top of already-tested
code.

SECURITY NOTES (read before exposing this beyond your own machine):
  - Binds to 127.0.0.1 (localhost only) by default. This app can execute
    REAL trades when ENABLE_LIVE_TRADING=true -- it has no authentication
    layer, so it must not be exposed on a network without adding one.
  - Flask's debug mode (interactive in-browser debugger) is OFF by default
    and should stay off in any environment where live trading could be
    enabled -- Werkzeug's debugger is a known code-execution risk if ever
    reachable by anyone other than you.
  - The live-trade confirmation flow never uses Python's input() (which
    would hang or crash in a web request with no attached terminal).
    Instead: /trade/confirm shows the exact order and requires typing the
    same confirmation phrase the CLI uses into a form field, plus a
    same-origin token minted for that specific page load. Both are
    validated server-side in /trade/execute before any order is placed.

Run with:
    python webapp/app.py
Then open http://127.0.0.1:5000
"""

import base64
import io
import json
import os
import re
import secrets
import sys
import tempfile
import threading
import time
import uuid

from flask import g, Flask, render_template, request, redirect, url_for, session, Response, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
import logging_utils
import pipeline
import visualize
import export_utils
import backtest as backtest_mod
from data_loader import InsufficientDataError
from data_sources.base import DataSourceError
from ensemble import AllModelsFailedError
from trading.trade_executor import execute_trade_decision, CONFIRMATION_PHRASE
from trading.signal_generator import generate_signal
from trading import trade_log
from webapp.api_v2 import bp as api_v2_bp
import forecast_service
import perf

log = logging_utils.get_logger("webapp")

app = Flask(__name__)
# `or`, not a getenv default: a BLANK value (copied from .env.example) is "" and used to leave Flask
# with no secret key, so every page returned 500.
app.secret_key = os.getenv("WEBAPP_SECRET_KEY") or secrets.token_hex(32)
# Explicit (Flask's own defaults are already reasonable, but stated
# explicitly here rather than relying on them staying that way):
# HTTPONLY blocks any client-side JS from reading the session cookie at
# all; SAMESITE=Lax blocks it being sent on cross-site requests. SECURE is
# deliberately left off -- this app binds to plain http://127.0.0.1, not
# https://, so a Secure-flagged cookie would just never be sent at all.
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Request-size cap. Every route here takes a short form post or a small JSON
# body -- nothing legitimately exceeds a few kilobytes. Without a cap, a
# single large POST can make the server buffer it in memory; Flask converts
# anything over this into a 413, handled below. This application has NO
# file-upload routes at all -- there is no multipart handling anywhere -- so
# this size cap is the entirety of its upload attack surface. A test asserts
# that this stays true (tests/test_hardening.py), so adding an upload route
# later trips it and forces proper extension/size/traversal handling.
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("WEBAPP_MAX_CONTENT_LENGTH_BYTES", 256 * 1024))


@app.after_request
def _security_headers(response):
    """
    Conservative headers appropriate for a self-contained local tool.

    Content-Security-Policy is deliberately permissive about inline styles
    and data: images -- the results page embeds its charts as
    <img src="data:image/png;base64,...">, and the templates carry inline
    style attributes, so a strict policy would visibly break the UI. It
    still blocks the things that matter here: no external script sources, no
    framing, no object embedding.
    """
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'",
    )
    # Never let a browser or proxy cache a page containing prediction state
    # or a live-trade confirmation.
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.errorhandler(413)
def _payload_too_large(e):
    return render_template("error.html",
                            message="That request was too large to process."), 413


# ---------------------------------------------------------------------------
# CSRF protection.
#
# This app is session/cookie-authenticated for its browser routes (the
# live-trading gate lives in the session, per `_operator_is_authenticated`),
# which is exactly the authentication model CSRF targets: a browser that
# already holds this app's session cookie can be made to fire a same-origin
# request from an attacker-controlled page, and the cookie goes along for
# the ride automatically. SESSION_COOKIE_SAMESITE="Lax" (set above) already
# blocks cross-site POST navigations in modern browsers, but that is a
# browser-version-dependent mitigation, not a guarantee -- so every
# state-changing browser route additionally requires an explicit,
# session-bound token, checked in constant time.
#
# Every GET stays completely unaffected. `/api/*` routes are deliberately
# EXCLUDED: they carry no session-based authentication or capability at all
# (no cookie is read or required to use them; they don't call
# _get_cached_result, touch `session`, or gate on `_operator_is_authenticated`
# the way the browser routes do), so there is no ambient browser credential
# for a forged cross-site request to ride on -- a CSRF token would protect
# nothing there that a curl script without a browser doesn't already bypass
# just as easily. This mirrors the "stateless API auth" carve-out: an /api/*
# caller authenticates (or doesn't) independently of this browser session.
# ---------------------------------------------------------------------------
def _get_csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_hex(32)
        session["csrf_token"] = token
    return token


def _csrf_token_valid(submitted) -> bool:
    expected = session.get("csrf_token")
    if not expected or not submitted:
        return False
    return secrets.compare_digest(str(submitted), str(expected))


@app.context_processor
def _inject_csrf_token():
    # Makes `csrf_token()` callable from every template without each route
    # having to pass it explicitly.
    return {"csrf_token": _get_csrf_token}


_CSRF_EXEMPT_PREFIXES = ("/api/",)


@app.before_request
def _perf_start():
    g._perf_t0 = time.perf_counter()


@app.after_request
def _perf_end(resp):
    t0 = getattr(g, "_perf_t0", None)
    if t0 is not None and request.endpoint and request.endpoint != "static":
        perf.record(f"http.{request.endpoint}", (time.perf_counter() - t0) * 1000)
    return resp


@app.before_request
def _throttle_expensive_legacy():
    """Legacy endpoints that train models (/predict, /api/predict, /api/backtest, ...) used to be unthrottled and,
    for /api/*, unauthenticated. They now share the per-IP limiter used by /api/v2."""
    if request.method == "POST" and not request.path.startswith("/api/v2") and (
            request.path.startswith("/api/") or request.path in ("/predict", "/backtest")):
        from webapp.api_v2 import check_rate
        return check_rate("legacy:" + request.path)
    return None


@app.before_request
def _enforce_csrf():
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    if request.path.startswith(_CSRF_EXEMPT_PREFIXES):
        return None
    submitted = request.form.get("csrf_token", "")
    if not _csrf_token_valid(submitted):
        return render_template(
            "error.html",
            message="Your session expired or this form was submitted from somewhere "
                    "unexpected, so it was rejected for your protection. Go back and "
                    "resubmit the form.",
        ), 403
    return None


# ---------------------------------------------------------------------------
# In-memory result store.
# This is a local, single-user tool (same trust model as the CLI it wraps),
# so a simple in-process dict keyed by a UUID in the session cookie is
# enough -- no database needed. Capped so a long-running dev server doesn't
# grow unbounded if left open for many predictions.
# ---------------------------------------------------------------------------
_RESULTS_CACHE = {}
_MAX_CACHE_ENTRIES = 20


_RESULTS_LOCK = threading.Lock()


def _store_result(payload: dict) -> str:
    result_id = uuid.uuid4().hex
    with _RESULTS_LOCK:         # written from the background worker, read/evicted from request threads
        _RESULTS_CACHE[result_id] = payload
        if len(_RESULTS_CACHE) > _MAX_CACHE_ENTRIES:
            oldest_key = min(_RESULTS_CACHE, key=lambda k: _RESULTS_CACHE[k]["created_at"])
            del _RESULTS_CACHE[oldest_key]
    return result_id


def _get_cached_result():
    result_id = session.get("result_id")
    if not result_id or result_id not in _RESULTS_CACHE:
        return None
    return _RESULTS_CACHE[result_id]


# ---------------------------------------------------------------------------
# Live-trading authentication.
#
# Everything else in this app is read-only analysis, so it keeps the CLI's
# trust model (localhost, single user). The live-trading routes are the one
# exception: they can move real money, so they require a shared secret that
# only the operator knows, on top of every gate that was already there
# (ENABLE_LIVE_TRADING in .env, the per-page session token, and the typed
# confirmation phrase).
#
# FAIL-CLOSED: with no WEBAPP_LIVE_TRADING_TOKEN configured -- the default --
# the live routes refuse outright. Live trading over HTTP therefore takes
# two independent, deliberate operator actions (ENABLE_LIVE_TRADING=true AND
# setting a token), neither of which any request to this app can perform.
#
# This is deliberately the smallest thing that actually holds rather than a
# full user/session/password system: adding real accounts would mean a user
# store, password hashing, and a login flow, which is a much larger change
# to a single-user local tool than the risk warrants. The tradeoff and its
# limits are documented in README.md's "Live trading safeguards" section.
# ---------------------------------------------------------------------------
def _live_trading_auth_state() -> tuple:
    """
    Returns (is_available, reason_if_not). `is_available` is True only when
    live trading is both enabled AND protected by an adequately long shared
    secret. The reason string is safe to show a user -- it never contains
    the token or any part of it.
    """
    if not config.ENABLE_LIVE_TRADING:
        return False, ("Live trading is disabled. Set ENABLE_LIVE_TRADING=true in your "
                       ".env file to enable it -- this is a deliberate step separate "
                       "from anything in this web form, by design.")
    token = config.WEBAPP_LIVE_TRADING_TOKEN
    if not token:
        return False, ("Live trading is enabled in .env but no WEBAPP_LIVE_TRADING_TOKEN "
                       "is set, so the live-trading routes are refusing to run. Set a "
                       "long random value for WEBAPP_LIVE_TRADING_TOKEN in .env and "
                       "restart the server.")
    if len(token) < config.WEBAPP_LIVE_TRADING_TOKEN_MIN_LENGTH:
        return False, (f"WEBAPP_LIVE_TRADING_TOKEN is shorter than the required "
                       f"{config.WEBAPP_LIVE_TRADING_TOKEN_MIN_LENGTH} characters, so it "
                       f"isn't providing meaningful protection. Live trading is refused "
                       f"until it is replaced with a longer random value.")
    return True, None


def _operator_is_authenticated() -> bool:
    """
    True when this session has already presented the correct shared secret.
    The secret itself is never stored in the session -- only the fact that
    it was verified.
    """
    return bool(session.get("live_trading_authenticated"))


def _check_operator_secret(submitted: str) -> bool:
    """
    Constant-time comparison of a submitted secret against the configured
    one. secrets.compare_digest avoids leaking how many leading characters
    matched via response timing.
    """
    expected = config.WEBAPP_LIVE_TRADING_TOKEN
    if not expected or not submitted:
        return False
    return secrets.compare_digest(str(submitted), str(expected))


def _build_votes(result: dict) -> list:
    """
    Builds the data for the "vote strip" -- each model's individual
    direction, using the exact same threshold rule signal_generator.py uses
    for the final call, so the individual votes and the final badge are
    always consistent with each other.
    """
    final = result["final_ensemble"]
    current_price = final["current_price"]
    components = final["component_predictions"]
    votes = []

    def direction_for(predicted_price):
        pct = (predicted_price - current_price) / current_price * 100
        if pct >= config.BUY_THRESHOLD_PCT:
            return "BUY", pct
        elif pct <= config.SELL_THRESHOLD_PCT:
            return "SELL", pct
        return "HOLD", pct

    if "linear_regression" in components:
        d, pct = direction_for(components["linear_regression"])
        votes.append({"name": "Linear Reg.", "direction": d, "pct_change": pct})
    if "lstm" in components:
        d, pct = direction_for(components["lstm"])
        votes.append({"name": "LSTM", "direction": d, "pct_change": pct})

    direction_map = {"up": "BUY", "down": "SELL", "flat": "HOLD"}
    for r in result["llm_results"]:
        if r.get("error") is None:
            raw_direction = (r["direction"] or "").lower()
            votes.append({
                "name": r["provider"],
                "direction": direction_map.get(raw_direction, "HOLD"),
                "pct_change": r["estimated_pct_change"],
            })
        else:
            votes.append({"name": r["provider"], "direction": "N/A", "pct_change": None})

    return votes


def _build_export_rows(result: dict) -> list:
    """
    Merges the LR and LSTM test-set (date, actual, predicted) series into
    one row-per-date table for CSV/PDF export. LR and LSTM use different
    windowing schemes so their test sets don't start on the same date or
    share the same length (documented in visualize.py) -- this merges by
    date string and leaves a blank where only one model has a value for
    that day, rather than forcing a shared index that would misalign them.
    """
    lr_rows = {
        str(d)[:10]: {"actual": float(a), "lr_predicted": float(p)}
        for d, a, p in zip(result["lr_dates_test"], result["y_test"], result["lr_eval"]["predictions"])
    }
    lstm_rows = {
        str(d)[:10]: float(p)
        for d, p in zip(result["lstm_dates_test"], result["lstm_eval"]["predictions"])
    }
    all_dates = sorted(set(lr_rows) | set(lstm_rows))

    rows = []
    for d in all_dates:
        lr_entry = lr_rows.get(d, {})
        rows.append({
            "date": d,
            "actual": lr_entry.get("actual"),
            "lr_predicted": lr_entry.get("lr_predicted"),
            "lstm_predicted": lstm_rows.get(d),
        })
    return rows


def _fig_to_base64(plot_fn, *args, **kwargs) -> str:
    """
    Calls one of visualize.py's existing save-to-path chart functions
    against a throwaway temp file, then reads it back as a base64 string
    for direct embedding in HTML (<img src="data:image/png;base64,...">).
    Keeps visualize.py itself untouched -- this is purely an adapter.

    Uses tempfile.mkstemp() (not a hand-built "/tmp/..." path) so the OS
    picks the actual temp directory (honors $TMPDIR, portable off Linux)
    and creates the file atomically/exclusively itself -- no window where a
    predictable path could be pre-created by something else first.
    """
    fd, tmp_path = tempfile.mkstemp(prefix="webapp_chart_", suffix=".png")
    os.close(fd)
    try:
        plot_fn(*args, save_path=tmp_path, **kwargs)
        with open(tmp_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return encoded
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.register_blueprint(api_v2_bp)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "workstation.html",
        default_ticker=config.TICKER,
        default_period=config.PERIOD,
        default_epochs=config.LSTM_EPOCHS,
        default_source=config.DATA_SOURCE,
    )


def _compute_legacy_predict(p: dict) -> str:
    """The expensive part of /predict. Runs on the background worker, NOT in the web request. Raises TaskError with
    the same user-facing messages/status codes the synchronous version returned."""
    ticker, run_backtest = p["ticker"], p["run_backtest"]
    try:
        result = pipeline.run_pipeline(
            ticker=ticker, period=p["period"], epochs=p["epochs"],
            source=p["source"], skip_llm=p["skip_llm"], verbose=False,
            walk_forward=p["walk_forward"],
            ensemble_weighting_mode="dynamic" if p["dynamic_weights"] else None,
            for_backtest=run_backtest,
        )
    except InsufficientDataError as e:
        raise forecast_service.TaskError(str(e), 400)
    except DataSourceError as e:
        raise forecast_service.TaskError(str(e), 400)
    except AllModelsFailedError as e:
        raise forecast_service.TaskError(str(e), 422)
    except Exception:
        # Deliberately generic: an unexpected exception could contain internals (paths, broker/API text). The real
        # exception is logged server-side.
        log.exception("Unexpected error in /predict")
        raise forecast_service.TaskError("Something went wrong while generating this prediction. Please try again.", 500)

    backtest_results = None
    if run_backtest:
        backtest_results = {}
        for model_name in ("lr", "lstm", "ensemble"):
            try:
                bt = backtest_mod.backtest_from_pipeline_result(result, model=model_name)
                backtest_results[model_name] = bt.to_dict()
            except ValueError:
                pass  # not enough overlapping data for this model -- skip it quietly

    chart_comparison = _fig_to_base64(
        visualize.plot_model_comparison,
        result["lr_dates_test"], result["y_test"], result["lr_eval"]["predictions"],
        result["lstm_dates_test"], result["lstm_eval"]["predictions"], ticker=ticker,
    )
    chart_panel = None
    if result["llm_results"]:
        chart_panel = _fig_to_base64(visualize.plot_llm_panel_votes, result["llm_results"])
    chart_ensemble = _fig_to_base64(visualize.plot_final_ensemble, result["final_ensemble"], ticker=ticker)

    votes = _build_votes(result)
    final_signal = generate_signal(result["final_ensemble"])
    export_rows = _build_export_rows(result)

    payload = {
        "created_at": time.time(),
        "ticker": ticker,
        "resolved_ticker": result.get("resolved_ticker"),
        "dropped_open_session_bar": result.get("dropped_open_session_bar"),
        "final_ensemble": result["final_ensemble"],
        "final_signal": final_signal,
        "votes": votes,
        "lr_eval": {k: v for k, v in result["lr_eval"].items() if k != "predictions"},
        "lstm_eval": {k: v for k, v in result["lstm_eval"].items()
                      if k not in ("predictions", "predictions_scaled", "actual",
                                   "actual_return", "predicted_return")},
        "llm_results": result["llm_results"],
        "llm_panel_summary": result["llm_panel_summary"],
        "walk_forward_lr": result.get("walk_forward_lr"),
        "walk_forward_lstm": result.get("walk_forward_lstm"),
        "confidence": result.get("confidence"),
        "data_quality": result.get("data_quality"),
        "run_metadata": result.get("run_metadata"),
        "backtest": backtest_results,
        "charts": {"comparison": chart_comparison, "panel": chart_panel, "ensemble": chart_ensemble},
        "export_rows": export_rows,
    }
    return _store_result(payload)


def _claim_predict_job(job: dict):
    """Turn a FINISHED job into the classic response. Only a job that actually succeeded can set the session's result
    id -- a failed/cancelled job never leaves a previous forecast on screen as if it were the new one."""
    status = job["status"]
    if status == "done":
        session["result_id"] = job["result_key"]
        return redirect(url_for("results"))
    session.pop("result_id", None)      # a failed/cancelled request must not leave an OLDER forecast showing as current
    if status == "cancelled":
        return render_template("error.html", message="This prediction was cancelled."), 409
    return render_template("error.html", message=job.get("error") or "The prediction failed."), int(job.get("http_status") or 500)


# One rule for every route that takes a ticker (letters/digits plus . - : & = and optional leading ^; up to 24 characters)
_TICKER_OK = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-:&=]{0,23}$")


@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.form.get("ticker", "").strip().upper()
    try:
        epochs = int(request.form.get("epochs", config.LSTM_EPOCHS))
    except ValueError:
        epochs = config.LSTM_EPOCHS
    if not ticker:
        return render_template("error.html", message="Please enter a ticker symbol."), 400
    if not _TICKER_OK.match(ticker):
        return render_template("error.html", message="That is not a valid ticker symbol. Use letters, digits and . - : & = "
                                                      "(optionally starting with ^), up to 24 characters."), 400
    params = {
        "ticker": ticker, "period": request.form.get("period", config.PERIOD),
        "source": request.form.get("source", config.DATA_SOURCE), "epochs": epochs,
        "skip_llm": request.form.get("skip_llm") == "on", "walk_forward": request.form.get("walk_forward") == "on",
        "dynamic_weights": request.form.get("dynamic_weights") == "on", "run_backtest": request.form.get("run_backtest") == "on",
    }
    # Identical in-flight requests share ONE job. The wait is bounded: if the work is not finished almost at once
    # (real LSTM training is not), the client gets 202 + a job id immediately and polls a real status endpoint.
    job = forecast_service.submit_task("legacy_predict", "legacy_predict|" + json.dumps(params, sort_keys=True),
                                       lambda j: _compute_legacy_predict(params))
    want_async = request.form.get("async") == "1"        # sent by the HTML forms: never hold a web worker
    if not want_async and forecast_service.wait_for(job, config.LEGACY_PREDICT_WAIT_SECONDS):
        return _claim_predict_job(job)
    return render_template("predict_wait.html", job_id=job["id"], timeout=int(config.JOB_TIMEOUT_SECONDS), ticker=ticker), 202


@app.route("/predict/claim/<job_id>", methods=["GET"])
def predict_claim(job_id):
    job = forecast_service.get_job(job_id)
    if not job or job.get("kind") != "legacy_predict":
        return render_template("error.html", message="Unknown or expired prediction job."), 404
    if job["status"] not in ("done", "failed", "cancelled"):
        return render_template("error.html", message="This prediction is still running."), 409
    return _claim_predict_job(job)


@app.route("/results", methods=["GET"])
def results():
    cached = _get_cached_result()
    if cached is None:
        return redirect(url_for("index"))
    live_available, live_reason = _live_trading_auth_state()
    return render_template(
        "results.html", r=cached,
        # Reflects the FULL gate (enabled in .env AND protected by an
        # adequately long operator secret), not just ENABLE_LIVE_TRADING --
        # otherwise the page would offer a live-trade form that can only
        # ever 401, which reads as a bug rather than as the safety control
        # it actually is.
        enable_live_trading=live_available,
        live_trading_blocked_reason=None if live_available else live_reason,
        live_trading_configured=config.ENABLE_LIVE_TRADING,
    )


@app.route("/export/csv", methods=["GET"])
def export_csv():
    cached = _get_cached_result()
    if cached is None:
        return redirect(url_for("index"))
    csv_text = export_utils.build_csv(cached)
    filename = f"{cached['ticker']}_prediction.csv"
    return Response(
        csv_text, mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/export/pdf", methods=["GET"])
def export_pdf():
    cached = _get_cached_result()
    if cached is None:
        return redirect(url_for("index"))
    try:
        pdf_bytes = export_utils.build_pdf(cached)
    except Exception as e:
        log.exception("PDF export failed")
        return render_template("error.html", message="Could not generate the PDF report."), 500
    filename = f"{cached['ticker']}_prediction.pdf"
    return Response(
        pdf_bytes, mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/trade", methods=["POST"])
def trade():
    cached = _get_cached_result()
    if cached is None:
        return redirect(url_for("index"))

    mode = request.form.get("mode", "paper")
    broker_name = request.form.get("broker", "paper")

    if mode == "live":
        # Never execute directly from this route -- always go through the
        # confirmation step first, regardless of what the form claims.
        available, reason = _live_trading_auth_state()
        if not available:
            return render_template("error.html", message=reason), 403
        if broker_name not in ("upstox", "groww"):
            return render_template("error.html",
                                    message="Live mode requires choosing Upstox or Groww as the broker."), 400
        # The operator secret is checked here, at the START of the live
        # flow, so an unauthenticated request never even reaches the
        # confirmation page that names the order.
        if not _operator_is_authenticated():
            if not _check_operator_secret(request.form.get("operator_secret", "")):
                return render_template(
                    "error.html",
                    message="Live trading requires the operator secret configured on this "
                            "server. Enter it in the live-trade form to continue.",
                ), 401
            session["live_trading_authenticated"] = True
        token = secrets.token_hex(16)
        session["live_trade_token"] = token
        session["live_trade_broker"] = broker_name
        return redirect(url_for("trade_confirm"))

    outcome = execute_trade_decision(
        ticker=cached["ticker"], final_ensemble=cached["final_ensemble"],
        mode="paper", broker_name="paper", auto_confirm=True, verbose=False,
    )
    return render_template("trade_result.html", outcome=outcome, ticker=cached["ticker"], mode="paper")


@app.route("/trade/confirm", methods=["GET"])
def trade_confirm():
    cached = _get_cached_result()
    token = session.get("live_trade_token")
    broker_name = session.get("live_trade_broker")
    if cached is None or not token or not broker_name:
        return redirect(url_for("index"))
    available, reason = _live_trading_auth_state()
    if not available:
        return render_template("error.html", message=reason), 403
    if not _operator_is_authenticated():
        return render_template(
            "error.html",
            message="Not authenticated for live trading. Start again from the results "
                    "page and enter the operator secret.",
        ), 401
    return render_template(
        "trade_confirm.html", r=cached, token=token, broker=broker_name,
        confirmation_phrase=CONFIRMATION_PHRASE,
    )


@app.route("/trade/execute", methods=["POST"])
def trade_execute():
    cached = _get_cached_result()
    expected_token = session.get("live_trade_token")
    broker_name = session.get("live_trade_broker")
    submitted_token = request.form.get("token", "")
    typed_phrase = request.form.get("confirmation_phrase", "")

    # Every one of these must independently hold, in this order, so a
    # missing/stale session or a resubmitted form can never slip through:
    if cached is None:
        return redirect(url_for("index"))
    available, reason = _live_trading_auth_state()
    if not available:
        return render_template("error.html", message=reason), 403
    if not _operator_is_authenticated():
        return render_template(
            "error.html",
            message="Not authenticated for live trading. Start again from the results "
                    "page and enter the operator secret.",
        ), 401
    if not expected_token or not secrets.compare_digest(str(submitted_token), str(expected_token)):
        return render_template("error.html",
                                message="This confirmation link has expired or was already used. "
                                        "Go back and try again."), 403
    if typed_phrase != CONFIRMATION_PHRASE:
        session.pop("live_trade_token", None)
        return render_template("trade_result.html", outcome={
            "executed": False,
            "reason": f"Confirmation phrase did not match exactly -- order cancelled.",
        }, ticker=cached["ticker"], mode="live")

    # Consume the token immediately so this exact confirmation can't be
    # replayed (e.g. by refreshing/resubmitting the POST).
    session.pop("live_trade_token", None)

    outcome = execute_trade_decision(
        ticker=cached["ticker"], final_ensemble=cached["final_ensemble"],
        mode="live", broker_name=broker_name, auto_confirm=True, verbose=False,
    )
    return render_template("trade_result.html", outcome=outcome, ticker=cached["ticker"], mode="live")


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", message="Something went wrong on the server."), 500


# ---------------------------------------------------------------------------
# JSON API (Phase 8)
# ---------------------------------------------------------------------------
# Thin JSON wrappers around the exact same pipeline.py / backtest.py used by
# the HTML routes above -- no prediction/trading logic is duplicated here.
# Every route validates its inputs and converts every expected failure mode
# (bad ticker, insufficient data, data-source outage, an internal error) into
# a structured JSON error response with an appropriate status code -- never
# a raw Python traceback, and never a process crash.
_VALID_API_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
_VALID_API_SOURCES = {"auto", "yfinance", "upstox", "groww"}
_VALID_API_MODELS = {"lr", "lstm", "ensemble"}


def _public_metrics(eval_dict: dict) -> dict:
    """
    Strips the bulky raw arrays out of a model's eval dict for JSON output,
    and exposes the directional-correctness metric under the unambiguous
    name `directional_accuracy_pct`.

    `accuracy_pct` is kept alongside it, unchanged, so existing API
    consumers and previously-saved report.json files keep working -- but the
    bare name "accuracy" is genuinely misleading here (it measures whether
    the predicted DIRECTION was right, not classification accuracy and not
    how close the predicted price was), so every new consumer should read
    `directional_accuracy_pct` instead.
    """
    drop = ("predictions", "predictions_scaled", "actual",
            "actual_return", "predicted_return")
    metrics = {k: v for k, v in (eval_dict or {}).items() if k not in drop}
    if "accuracy_pct" in metrics:
        metrics["directional_accuracy_pct"] = metrics["accuracy_pct"]
    return metrics


def _api_request_data() -> dict:
    """Accepts either a JSON body or form-encoded data, so /api/* is usable
    from curl -d '{"ticker":"AAPL"}' -H 'Content-Type: application/json' or
    a plain HTML form post."""
    json_body = request.get_json(silent=True)
    if isinstance(json_body, dict):
        return json_body
    return request.form.to_dict()




def _validate_ticker(raw) -> tuple:
    """Returns (ticker, error_response_or_None)."""
    if not raw or not isinstance(raw, str) or not raw.strip():
        return None, (jsonify({"error": "invalid_ticker",
                                "message": "'ticker' is required and must be a non-empty string."}), 400)
    ticker = raw.strip().upper()
    if not _TICKER_OK.match(ticker):
        return None, (jsonify({"error": "invalid_ticker",
                                "message": "'ticker' contains characters that aren't a valid symbol."}), 400)
    return ticker, None


def _validate_period(raw) -> tuple:
    if raw is None or raw == "":
        return config.PERIOD, None
    if raw not in _VALID_API_PERIODS:
        return None, (jsonify({"error": "invalid_period",
                                "message": f"'period' must be one of {sorted(_VALID_API_PERIODS)}."}), 400)
    return raw, None


def _validate_epochs(raw) -> tuple:
    if raw is None or raw == "":
        return config.LSTM_EPOCHS, None
    try:
        epochs = int(raw)
    except (TypeError, ValueError):
        return None, (jsonify({"error": "invalid_epochs", "message": "'epochs' must be an integer."}), 400)
    if not (1 <= epochs <= 500):
        return None, (jsonify({"error": "invalid_epochs",
                                "message": "'epochs' must be between 1 and 500."}), 400)
    return epochs, None


def _validate_source(raw) -> tuple:
    if raw is None or raw == "":
        return config.DATA_SOURCE, None
    if raw not in _VALID_API_SOURCES:
        return None, (jsonify({"error": "invalid_source",
                                "message": f"'source' must be one of {sorted(_VALID_API_SOURCES)}."}), 400)
    return raw, None


def _run_pipeline_for_api(ticker, period, epochs, source, skip_llm, walk_forward=False,
                           ensemble_weighting_mode=None, for_backtest=False):
    """
    Shared error handling for every /api/* route that runs the pipeline:
    converts every known failure mode into a structured (response, status)
    pair, and never lets an unexpected exception escape as a raw traceback.
    Returns (result_dict_or_None, error_response_or_None).
    """
    try:
        result = pipeline.run_pipeline(
            ticker=ticker, period=period, epochs=epochs, source=source,
            skip_llm=skip_llm, verbose=False, walk_forward=walk_forward,
            ensemble_weighting_mode=ensemble_weighting_mode, for_backtest=for_backtest,
        )
        return result, None
    except InsufficientDataError as e:
        return None, (jsonify({"error": "insufficient_data", "message": str(e)}), 422)
    except DataSourceError as e:
        return None, (jsonify({"error": "data_source_error", "message": str(e)}), 502)
    except AllModelsFailedError as e:
        return None, (jsonify({"error": "all_models_failed", "message": str(e)}), 422)
    except Exception:
        # Deliberately generic: never echo internal exception text (could
        # contain file paths or other internals) back to an API consumer.
        return None, (jsonify({"error": "internal_error",
                                "message": "Prediction failed unexpectedly. Check server logs."}), 500)


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "service": "stock-prediction-ensemble"})


def _pipeline_or_task_error(**kw):
    """Run the pipeline on the background worker; map every failure to a TaskError carrying the API's error code/status."""
    try:
        return pipeline.run_pipeline(verbose=False, **kw)
    except InsufficientDataError as e:
        raise forecast_service.TaskError(str(e), 422, "insufficient_data")
    except DataSourceError as e:
        raise forecast_service.TaskError(str(e), 502, "data_source_error")
    except AllModelsFailedError as e:
        raise forecast_service.TaskError(str(e), 422, "all_models_failed")
    except Exception:
        log.exception("Unexpected error in a background API computation")
        raise forecast_service.TaskError("Prediction failed unexpectedly. Check server logs.", 500, "internal_error")


def _compute_api_predict(ticker, period, epochs, source, skip_llm, walk_forward, dynamic_weights, run_backtest_after, job):
    """Background half of /api/predict: returns the JSON-able body (stored on the job); raises TaskError(code=...)."""
    result = _pipeline_or_task_error(
        ticker=ticker, period=period, epochs=epochs, source=source, skip_llm=skip_llm, walk_forward=walk_forward,
        ensemble_weighting_mode="dynamic" if dynamic_weights else None, for_backtest=run_backtest_after)
    signal = generate_signal(result["final_ensemble"])
    job["_payload"] = {
        "ticker": ticker, "current_price": result["current_price"], "data_quality": result["data_quality"],
        "linear_regression": {"next_day_prediction": result["lr_next_price"], "metrics": _public_metrics(result["lr_eval"])},
        "lstm": {"next_day_prediction": result["lstm_next_price"], "metrics": _public_metrics(result["lstm_eval"])},
        "llm_panel": result["llm_panel_summary"], "final_ensemble": result["final_ensemble"],
        "confidence": result["confidence"], "signal": signal,
        "walk_forward": {"linear_regression": result["walk_forward_lr"], "lstm": result["walk_forward_lstm"]},
        "run_metadata": result["run_metadata"]}
    return "api_predict"


def _compute_api_signal(ticker, period, epochs, job):
    result = _pipeline_or_task_error(ticker=ticker, period=period, epochs=epochs, source=config.DATA_SOURCE, skip_llm=True)
    job["_payload"] = {"ticker": ticker, "signal": generate_signal(result["final_ensemble"]),
                       "final_ensemble": result["final_ensemble"], "confidence": result["confidence"]}
    return "api_signal"


def _compute_api_backtest(ticker, period, epochs, model, weighting_mode, dynamic_weights, job):
    result = _pipeline_or_task_error(ticker=ticker, period=period, epochs=epochs, source=config.DATA_SOURCE, skip_llm=True,
                                     ensemble_weighting_mode="dynamic" if dynamic_weights else None, for_backtest=True)
    try:
        bt = backtest_mod.backtest_from_pipeline_result(result, model=model, weighting_mode=weighting_mode)
    except ValueError as e:
        raise forecast_service.TaskError(str(e), 422, "backtest_error")
    except Exception:
        log.exception("Unexpected error in /api/backtest")
        raise forecast_service.TaskError("Backtest failed unexpectedly. Check server logs.", 500, "internal_error")
    job["_payload"] = bt.to_dict()
    return "api_backtest"


def _api_dispatch(kind, key_parts, compute, data):
    """Background job with a bounded wait. `async` -> immediate 202 + job id; otherwise the old synchronous contract
    within LEGACY_PREDICT_WAIT_SECONDS, then 202. Identical in-flight requests share one job."""
    job = forecast_service.submit_task(kind, kind + "|" + json.dumps(key_parts, sort_keys=True), compute)
    want_async = str(data.get("async", "")).lower() in ("1", "true", "on", "yes")
    if want_async or not forecast_service.wait_for(job, config.LEGACY_PREDICT_WAIT_SECONDS):
        return jsonify({"status": job["status"] if job["status"] in ("queued", "running") else "running", "job_id": job["id"],
                        "status_url": f"/api/v2/jobs/{job['id']}", "result_url": f"/api/jobs/{job['id']}/result"}), 202
    return _api_job_response(job)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = _api_request_data()
    ticker, err = _validate_ticker(data.get("ticker"))
    if err:
        return err
    period, err = _validate_period(data.get("period"))
    if err:
        return err
    epochs, err = _validate_epochs(data.get("epochs"))
    if err:
        return err
    source, err = _validate_source(data.get("source"))
    if err:
        return err
    truthy = lambda k: str(data.get(k, "")).lower() in ("1", "true", "on", "yes")
    args = (ticker, period, epochs, source, truthy("skip_llm"), truthy("walk_forward"), truthy("dynamic_weights"), truthy("for_backtest"))
    return _api_dispatch("api_predict", args, lambda j: _compute_api_predict(*args, j), data)


def _api_job_response(job):
    if job["status"] == "done":
        return jsonify(job["_payload"])
    if job["status"] == "cancelled":
        return jsonify({"error": "cancelled", "message": "The prediction was cancelled."}), 409
    return jsonify({"error": job.get("error_code") or "internal_error", "message": job.get("error") or "Prediction failed."}), int(job.get("http_status") or 500)


_API_JOB_KINDS = ("api_predict", "api_signal", "api_backtest")


@app.route("/api/jobs/<job_id>/result", methods=["GET"])
@app.route("/api/predict/result/<job_id>", methods=["GET"])        # V18 URL, kept
def api_job_result(job_id):
    job = forecast_service.get_job(job_id)
    if not job or job.get("kind") not in _API_JOB_KINDS:
        return jsonify({"error": "not_found", "message": "Unknown or expired job."}), 404
    if job["status"] not in ("done", "failed", "cancelled"):
        return jsonify({"status": job["status"], "stage": job["stage"]}), 202
    return _api_job_response(job)


@app.route("/api/signal", methods=["POST"])
def api_signal():
    """Lighter than /api/predict: skips the LLM panel and walk-forward by default, for a fast BUY/SELL/HOLD call."""
    data = _api_request_data()
    ticker, err = _validate_ticker(data.get("ticker"))
    if err:
        return err
    period, err = _validate_period(data.get("period"))
    if err:
        return err
    epochs, err = _validate_epochs(data.get("epochs"))
    if err:
        return err
    return _api_dispatch("api_signal", (ticker, period, epochs), lambda j: _compute_api_signal(ticker, period, epochs, j), data)


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    data = _api_request_data()
    ticker, err = _validate_ticker(data.get("ticker"))
    if err:
        return err
    period, err = _validate_period(data.get("period"))
    if err:
        return err
    epochs, err = _validate_epochs(data.get("epochs"))
    if err:
        return err
    model = data.get("model") or "ensemble"
    if model not in _VALID_API_MODELS:
        return jsonify({"error": "invalid_model",
                         "message": f"'model' must be one of {sorted(_VALID_API_MODELS)}."}), 400
    weighting_mode = data.get("weighting_mode") or "auto"
    if weighting_mode not in ("auto", "fixed", "dynamic"):
        return jsonify({"error": "invalid_weighting_mode",
                         "message": "'weighting_mode' must be one of ['auto', 'fixed', 'dynamic']."}), 400
    dynamic_weights = str(data.get("dynamic_weights", "")).lower() in ("1", "true", "on", "yes")
    args = (ticker, period, epochs, model, weighting_mode, dynamic_weights)
    return _api_dispatch("api_backtest", args, lambda j: _compute_api_backtest(*args, j), data)


if __name__ == "__main__":
    logging_utils.configure()
    if config.WARMUP_ON_START:                       # background: never delays the server from accepting requests
        threading.Thread(target=forecast_service.warm_up, name="warmup", daemon=True).start()
    debug_mode = os.getenv("WEBAPP_DEBUG", "false").lower() == "true"
    # HARD REFUSAL, not a warning: Werkzeug's debugger is arbitrary code
    # execution for anyone who can reach it. Combined with a server that can
    # place real broker orders, that is not a risk to leave to a printed
    # caution the operator may scroll past.
    if debug_mode and config.ENABLE_LIVE_TRADING:
        raise SystemExit(
            "Refusing to start: WEBAPP_DEBUG=true and ENABLE_LIVE_TRADING=true are set "
            "together. Werkzeug's interactive debugger allows arbitrary code execution "
            "for anyone who can reach this server, and this server can place real "
            "orders. Turn one of them off."
        )
    port = int(os.getenv("WEBAPP_PORT", 5000))
    print(f"\nStarting on http://127.0.0.1:{port}  (localhost only)")
    if config.ENABLE_LIVE_TRADING:
        print("!!! ENABLE_LIVE_TRADING=true -- this server can place REAL orders. !!!")
    app.run(host="127.0.0.1", port=port, debug=debug_mode)
