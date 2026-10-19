"""
webapp/api_v2.py -- JSON API used by the workstation UI.

Security model
  * GET routes are read-only. They never return a secret (only booleans like "key configured").
  * Every state-changing route (POST) requires the session-bound token in the `X-CSRF-Token` header
    AND, if the browser sent an Origin header, that origin must match this host.
  * Expensive routes are rate limited per client IP (config.WEBAPP_EXPENSIVE_RPM).
  * No route here can place a LIVE order. Paper orders only, priced from a freshly retrieved quote.
  * Provider/upstream error text is always passed through security.redact and truncated.
"""
import secrets
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from flask import Blueprint, jsonify, request, session

import config
import security
import market_data
import indicators
import forecast_service
import news_feed
import research
import provider_verification
import perf
import logging_utils
from ai_integrations import providers

bp = Blueprint("api_v2", __name__, url_prefix="/api/v2")
_log = logging_utils.get_logger("api_v2")
API_VERSION = "17.0"

_hits = defaultdict(deque)
_hits_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="quotes")


@bp.after_request
def _no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _err(code, message, status):
    return jsonify({"error": code, "message": security.redact(message)}), status


def csrf_token():
    t = session.get("csrf_token")
    if not t:
        t = secrets.token_hex(32)
        session["csrf_token"] = t
    return t


def state_changing(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        sent, expected = request.headers.get("X-CSRF-Token", ""), session.get("csrf_token")
        if not expected or not sent or not secrets.compare_digest(sent, expected):
            return _err("csrf", "Missing or invalid CSRF token. Reload the page.", 403)
        origin = request.headers.get("Origin")
        if origin and origin.split("://", 1)[-1] != request.host:
            return _err("csrf", "Cross-origin request refused.", 403)
        return fn(*a, **kw)
    return wrapper


def check_rate(bucket: str):
    """Sliding 60 s window per (client IP, bucket). Returns a 429 response, or None if allowed."""
    ip, now = request.remote_addr or "?", time.monotonic()
    with _hits_lock:
        q = _hits[(ip, bucket)]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= config.WEBAPP_EXPENSIVE_RPM:
            retry = max(1, int(60 - (now - q[0])))
            r = jsonify({"error": "rate_limited", "message": f"Too many requests. Retry in {retry}s."})
            r.status_code, r.headers["Retry-After"] = 429, str(retry)
            return r
        q.append(now)
    return None


def rate_limited(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        return check_rate(fn.__name__) or fn(*a, **kw)
    return wrapper


def _symbol():
    s = request.args.get("symbol") or (request.get_json(silent=True) or {}).get("symbol") or ""
    try:
        return s.strip(), market_data.resolve_instrument(s)
    except market_data.InvalidSymbol as e:
        raise _Bad(str(e))


class _Bad(Exception):
    pass


@bp.errorhandler(_Bad)
def _bad(e):
    return _err("invalid_input", str(e), 400)


@bp.errorhandler(ValueError)
def _val(e):
    return _err("invalid_input", str(e), 400)


@bp.errorhandler(Exception)
def _unexpected(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return _err(e.name.lower().replace(" ", "_"), e.description or e.name, e.code)
    _log.exception("api_v2 error")                 # project logger: redaction filter covers the traceback
    return _err("internal_error", "Unexpected server error. See server logs.", 500)


# ---------------------------------------------------------------- meta / health
@bp.get("/meta")
def meta():
    live_ok = bool(config.ENABLE_LIVE_TRADING)
    return jsonify({"api_version": API_VERSION, "csrf_token": csrf_token(),
                    "trading": {"default_mode": "paper", "live_trading_enabled_in_env": live_ok,
                                "live_orders_via_this_api": False},
                    "limits": {"freshness_live_max_age_seconds": config.LIVE_MAX_AGE_SECONDS,
                               "quote_ttl_seconds": config.QUOTE_TTL_SECONDS},
                    "server_time_utc": market_data.datetime.now(market_data.timezone.utc).isoformat()})


@bp.get("/health")
def health():
    import checkpoints
    ai = providers.get_registry().statuses()
    jobs = [forecast_service.public_job(j) for j in list(forecast_service.JOBS.values())[-10:]]
    return jsonify({"market_data": market_data.get_service().health(), "ai": ai, "performance": perf.snapshot(),
                    "jobs": [{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs],
                    "checkpoints": checkpoints.list_checkpoints(),
                    "forecast_cache_entries": len(forecast_service._RESULTS),
                    "news_feeds_configured": len(news_feed.configured_feeds()),
                    "verification": {**provider_verification.load(),
                                     "note": "Written only by `python scripts/verify_providers.py` on the machine that ran it. "
                                             "If empty, NOTHING has been verified against a real provider."}})


# ---------------------------------------------------------------- market data
@bp.get("/quote")
def quote():
    raw, _ = _symbol()
    force = request.args.get("force") == "1"
    return jsonify(market_data.get_service().get_quote(raw, force=force).to_dict())


@bp.get("/quotes")
def quotes():
    syms = [s for s in (request.args.get("symbols", "")).split(",") if s.strip()][:12]
    if not syms:
        raise _Bad("symbols is required (comma separated, max 12)")
    svc = market_data.get_service()
    def one(s):
        try:
            return svc.get_quote(s.strip()).to_dict()
        except market_data.InvalidSymbol as e:
            return {"symbol": s, "freshness": "unavailable", "error": str(e)}
    return jsonify({"quotes": list(_pool.map(one, syms))})


@bp.get("/history")
def history():
    raw, _ = _symbol()
    period = request.args.get("period", "1y")
    try:
        h = market_data.get_service().get_history(raw, period)
    except market_data.ProviderError as e:
        return _err("data_unavailable", str(e), 502)
    except market_data.RateLimited as e:
        return _err("rate_limited", str(e), 429)
    ind = indicators.compute(h["bars"])
    return jsonify({**h, "indicators": ind})


@bp.get("/news")
def news():
    raw, inst = _symbol()
    return jsonify(news_feed.get_news(inst.yahoo_symbol))


# ---------------------------------------------------------------- forecast jobs
def _job_json(job):
    j = forecast_service.public_job(job)
    return {k: v for k, v in j.items() if not k.startswith("_")}


@bp.post("/forecast")
@state_changing
@rate_limited
def forecast_submit():
    body = request.get_json(silent=True) or {}
    raw = (body.get("symbol") or "").strip()
    market_data.resolve_instrument(raw)
    try:
        job = forecast_service.submit_forecast(raw, body.get("period", "2y"), body.get("epochs"),
                                               walk_forward=bool(body.get("walk_forward")))
    except market_data.ProviderError as e:
        return _err("data_unavailable", str(e), 502)
    except market_data.RateLimited as e:
        return _err("rate_limited", str(e), 429)
    return jsonify(_job_json(job)), 202


@bp.get("/jobs/<job_id>")
def job_status(job_id):
    job = forecast_service.get_job(job_id)
    if not job:
        return _err("not_found", "Unknown job.", 404)
    return jsonify(_job_json(job))


@bp.post("/jobs/<job_id>/cancel")
@state_changing
def job_cancel(job_id):
    return jsonify({"cancel_requested": forecast_service.cancel_job(job_id),
                    "note": "Cancellation is checked between stages; a model fit already running finishes first."})


@bp.get("/forecast/result")
def forecast_result():
    stash = forecast_service.get_result(request.args.get("key", ""))
    if not stash:
        return _err("not_found", "No cached forecast for this key (it may have expired). Run it again.", 404)
    return jsonify({"key": request.args["key"], **stash["payload"]})


@bp.post("/backtest")
@state_changing
@rate_limited
def backtest():
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(forecast_service.run_backtest_job(body.get("key", ""), body))
    except LookupError as e:
        return _err("not_found", str(e), 404)


# ---------------------------------------------------------------- AI
@bp.get("/ai/status")
def ai_status():
    return jsonify({"providers": providers.get_registry().statuses(),
                    "note": "'verified' means a real request succeeded. A configured key alone is shown as 'untested'."})


@bp.post("/ai/test")
@state_changing
@rate_limited
def ai_test():
    pid = (request.get_json(silent=True) or {}).get("provider", "")
    p = providers.get_registry().providers.get(pid)
    if not p:
        raise _Bad("Unknown provider id.")
    return jsonify(p.test_connection())


@bp.post("/ai/research")
@state_changing
@rate_limited
def ai_research():
    body = request.get_json(silent=True) or {}
    stash = forecast_service.get_result(body.get("key", ""))
    if not stash:
        return _err("not_found", "Run a forecast first; AI commentary explains computed results only.", 404)
    p = stash["payload"]
    inst = p["instrument"]
    svc = market_data.get_service()
    q = svc.get_quote(inst["yahoo_symbol"]).to_dict()
    n = news_feed.get_news(inst["yahoo_symbol"])
    hist = svc.get_history(inst["yahoo_symbol"], "1y")
    pack = research.build_fact_pack(
        instrument={"symbol": inst["symbol"], "exchange": inst["exchange"]}, quote=q,
        forecast={"components": p["components"], "final_price": p["final_price"], "pct_change": p["pct_change"],
                  "as_of_bar": p["as_of_bar"], "target": p["target"]["lstm"], "interval": p["interval"]},
        evaluation={"models": p["evaluation"]["models"], "period": p["evaluation"]["period"], "n_obs": p["evaluation"]["n_obs"]},
        news=n["items"], indicators=indicators.compute(hist["bars"])["latest"])
    out = research.generate(pack, force=bool(body.get("force")))
    out["news_items"] = n["items"][:8]
    return jsonify(out)


# ---------------------------------------------------------------- paper portfolio & risk
def _broker():
    from trading.brokers.paper_broker import PaperBroker
    return PaperBroker()


def _portfolio_snapshot():
    b = _broker()
    svc = market_data.get_service()
    positions, equity = [], b.get_available_capital()
    for sym, qty in b.state["holdings"].items():
        if qty <= 0:
            continue
        try:
            q = svc.get_quote(sym).to_dict()
        except Exception as e:
            q = {"price": None, "freshness": "unavailable", "source_ts": None, "provider": "unknown", "error": str(e)}
        px = q["price"] if q.get("price") and q["freshness"] != "unavailable" else None
        avg = b.get_avg_entry_price(sym)
        val = qty * px if px is not None else None
        if val is not None:
            equity += val
        positions.append({"symbol": sym, "quantity": qty, "avg_entry": avg, "price": px, "market_value": val,
                          "unrealized_pnl": (px - avg) * qty if px and avg else None,
                          "quote_freshness": q["freshness"], "quote_source_ts": q.get("source_ts"),
                          "quote_provider": q.get("provider")})
    return b, positions, equity


@bp.get("/portfolio")
def portfolio():
    b, positions, equity = _portfolio_snapshot()
    from trading.trade_log import count_trades_today
    return jsonify({"kind": "PAPER (simulated) -- not a real account", "cash": b.get_available_capital(),
                    "starting_cash_config": config.PAPER_STARTING_CASH, "positions": positions,
                    "equity_estimate": equity,
                    "equity_note": "Positions with no usable quote are excluded from equity.",
                    "trades_today": count_trades_today(), "retrieved_ts": market_data.datetime.now(market_data.timezone.utc).isoformat(),
                    "live_account": {"connected_via_this_ui": False,
                                     "message": "Real broker balances are not displayed here. Live orders use the separate, gated classic flow."}})


@bp.post("/paper/reset")
@state_changing
def paper_reset():
    import os
    body = request.get_json(silent=True) or {}
    try:
        cash = float(body.get("starting_cash"))
    except (TypeError, ValueError):
        raise _Bad("starting_cash must be a number")
    if not (1 <= cash <= 1e12) or body.get("confirm") is not True:
        raise _Bad("starting_cash must be between 1 and 1e12 and confirm must be true")
    path = config.PAPER_PORTFOLIO_PATH
    if os.path.exists(path):
        os.remove(path)
    from trading.brokers.paper_broker import PaperBroker
    PaperBroker(starting_cash=cash)
    return jsonify({"ok": True, "starting_cash": cash})


@bp.post("/paper/order")
@state_changing
@rate_limited
def paper_order():
    from trading import risk_manager, trade_log
    body = request.get_json(silent=True) or {}
    raw, inst = _symbol()
    side = str(body.get("side", "")).upper()
    try:
        qty = int(body.get("quantity"))
    except (TypeError, ValueError):
        raise _Bad("quantity must be a whole number")
    if side not in ("BUY", "SELL") or qty <= 0 or body.get("confirm") is not True:
        raise _Bad("side must be BUY or SELL, quantity positive, and confirm must be true")
    q = market_data.get_service().get_quote(raw, force=True)
    if q.price is None or q.freshness in ("unavailable", "stale") or "served_cached_during_outage" in q.quality_flags:
        return _err("no_fresh_quote", f"No usable quote ({q.freshness}); a simulated order needs a current price. "
                    f"{q.error or ''}", 409)
    b = _broker()
    try:
        risk_manager.check_daily_trade_limit()
        held = b.get_position(inst.yahoo_symbol)
        risk_manager.validate_order(qty, q.price, b.get_available_capital(), action=side, held_quantity=held)
    except (risk_manager.RiskLimitError, ValueError) as e:
        trade_log.log_trade(ticker=inst.yahoo_symbol, action=side, mode="paper", broker="paper",
                            predicted_pct_change=0.0, current_price=q.price, predicted_price=q.price,
                            quantity=qty, order_value=qty * q.price, status="BLOCKED", message=str(e))
        return _err("risk_blocked", str(e), 422)
    res = b.place_order(inst.yahoo_symbol, side, qty, price=q.price)
    trade_log.log_trade(ticker=inst.yahoo_symbol, action=side, mode="paper", broker="paper",
                        predicted_pct_change=0.0, current_price=q.price, predicted_price=q.price,
                        quantity=qty, order_value=qty * q.price, status=res["status"],
                        broker_order_id=res["broker_order_id"], message=res["message"])
    return jsonify({"simulated": True, "result": res, "price_used": q.price, "quote_freshness": q.freshness,
                    "quote_source_ts": q.source_ts, "quote_provider": q.provider})


@bp.get("/risk")
def risk():
    import numpy as np
    from trading.trade_log import count_trades_today
    b, positions, equity = _portfolio_snapshot()
    svc = market_data.get_service()
    for p in positions:
        p["weight_pct"] = (p["market_value"] / equity * 100) if p["market_value"] and equity else None
        try:
            bars = svc.get_history(p["symbol"], "1y")["bars"]
            c = np.array([x["c"] for x in bars])
            r = np.diff(c) / c[:-1]
            dd = (c / np.maximum.accumulate(c) - 1).min()
            p["vol_annualized_pct"] = float(r.std(ddof=1) * np.sqrt(252) * 100) if len(r) > 20 else None
            p["max_drawdown_1y_pct"] = float(dd * 100)
        except Exception:  # noqa: BLE001 - risk page must still render
            p["vol_annualized_pct"] = p["max_drawdown_1y_pct"] = None
    return jsonify({"equity_estimate": equity, "cash": b.get_available_capital(),
                    "cash_weight_pct": (b.get_available_capital() / equity * 100) if equity else None,
                    "largest_position_pct": max([p["weight_pct"] or 0 for p in positions], default=0),
                    "positions": positions,
                    "limits": {"max_position_pct_of_capital": config.MAX_POSITION_PCT * 100,
                               "max_order_value": config.MAX_ORDER_VALUE, "max_trades_per_day": config.MAX_TRADES_PER_DAY,
                               "stop_loss_pct": config.STOP_LOSS_PCT or None, "take_profit_pct": config.TAKE_PROFIT_PCT or None,
                               "max_portfolio_exposure_pct": config.MAX_PORTFOLIO_EXPOSURE_PCT or None,
                               "daily_loss_limit_pct": config.DAILY_LOSS_LIMIT_PCT or None},
                    "trades_today": count_trades_today(),
                    "note": "Paper portfolio only. Volatility and drawdown are trailing 1y, from adjusted daily closes."})


@bp.get("/risk/size")
def risk_size():
    from trading import risk_manager
    raw, _ = _symbol()
    try:
        risk_pct, stop_pct = float(request.args["risk_pct"]), float(request.args["stop_pct"])
    except (KeyError, ValueError):
        raise _Bad("risk_pct and stop_pct (percent numbers) are required")
    if not (0 < risk_pct <= 10 and 0 < stop_pct <= 50):
        raise _Bad("risk_pct must be in (0,10] and stop_pct in (0,50]")
    q = market_data.get_service().get_quote(raw)
    if q.price is None or q.price <= 0:
        return _err("no_quote", f"No positive quote available ({q.price}): {q.error or ''}", 409)
    b, _, equity = _portfolio_snapshot()
    denom = q.price * stop_pct / 100
    by_risk = int(max(0.0, equity) * risk_pct / 100 // denom) if denom > 0 else 0
    cap = risk_manager.compute_order_quantity(max(0.0, b.get_available_capital()), q.price)
    return jsonify({"price": q.price, "quote_freshness": q.freshness, "equity": equity,
                    "quantity_by_risk": by_risk, "quantity_cap_by_limits": cap, "suggested_quantity": max(0, min(by_risk, cap)),
                    "formula": "floor(equity * risk% / (price * stop%)), then capped by MAX_POSITION_PCT / MAX_ORDER_VALUE / cash"})


# ---------------------------------------------------------------- settings (read-only)
@bp.get("/settings")
def settings():
    c = config
    return jsonify({
        "editable_in": ".env (restart required). Secrets are never shown or editable here.",
        "data": {"sources": ["yahoo (yfinance)", "upstox (NSE/BSE quotes, needs daily token)", "groww (history fallback)"],
                 "upstox_configured": bool(c.UPSTOX_CLIENT_ID), "groww_configured": bool(c.GROWW_ACCESS_TOKEN or c.GROWW_API_KEY),
                 "quote_ttl_seconds": c.QUOTE_TTL_SECONDS, "history_ttl_seconds": c.HISTORY_TTL_SECONDS,
                 "live_max_age_seconds": c.LIVE_MAX_AGE_SECONDS},
        "models": {"lookback": c.LOOKBACK_WINDOW, "lstm_target": c.LSTM_PREDICTION_TARGET, "lstm_units": c.LSTM_UNITS,
                   "lstm_epochs": c.LSTM_EPOCHS, "train_test_split": c.TRAIN_TEST_SPLIT, "seed": c.RANDOM_SEED,
                   "ensemble_weights": c.ENSEMBLE_WEIGHTS, "llm_weight_note": c.LLM_WEIGHT_NOTE},
        "risk": {"max_position_pct": c.MAX_POSITION_PCT * 100, "max_order_value": c.MAX_ORDER_VALUE,
                 "max_trades_per_day": c.MAX_TRADES_PER_DAY, "paper_starting_cash": c.PAPER_STARTING_CASH},
        "trading": {"mode": "paper", "live_trading_enabled_in_env": bool(c.ENABLE_LIVE_TRADING),
                    "live_token_configured": bool(c.WEBAPP_LIVE_TRADING_TOKEN)},
        "ai": {"providers": [{"id": p["id"], "key_env": p["key_env"], "model_env": p["model_env"],
                              "key_configured": p["state"] != "missing_key", "model": p["model"]}
                             for p in providers.get_registry().statuses()]},
        "news": {"feeds_configured": len(news_feed.configured_feeds()), "env": "NEWS_FEEDS"}})
