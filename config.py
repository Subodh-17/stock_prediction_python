"""
config.py
---------
Central configuration for the Multi-AI Stock Prediction Ensemble project.

All API keys are read from environment variables (or a local .env file, which
python-dotenv will load automatically). NEVER hard-code real API keys into
this file or into version control.

To set up:
1. Copy `.env.example` to `.env`
2. Fill in whichever keys you actually have. You do NOT need all five LLM
   keys -- the ensemble will simply skip any provider whose key is missing
   and rely on Linear Regression + LSTM + whichever LLMs are configured.
"""

import os
from dotenv import load_dotenv

import _native_preload
_native_preload.preload()   # MUST run before anything can import TensorFlow -- see _native_preload.py

load_dotenv()  # loads variables from a .env file if present

# ---------------------------------------------------------------------------
# Stock / data settings
# ---------------------------------------------------------------------------
TICKER = os.getenv("TICKER", "AAPL")          # default ticker symbol
PERIOD = os.getenv("PERIOD", "2y")            # how much history to pull
INTERVAL = os.getenv("INTERVAL", "1d")        # daily candles
LOOKBACK_WINDOW = int(os.getenv("LOOKBACK_WINDOW", 30))   # days fed into LSTM
TRAIN_TEST_SPLIT = float(os.getenv("TRAIN_TEST_SPLIT", 0.8))

# ---------------------------------------------------------------------------
# LSTM hyperparameters
# ---------------------------------------------------------------------------
LSTM_UNITS = int(os.getenv("LSTM_UNITS", 64))
LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", 25))
LSTM_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", 16))

# What the LSTM predicts: "return" (default) trains it to map a window of
# past daily % returns to the next day's % return, then reconstructs a price
# as base_price * (1 + predicted_return). "price" trains it to map a window
# of past raw (MinMax-scaled) prices directly to the next raw price.
#
# This defaults to "return" based on an actual empirical comparison run
# during development (see README's "LSTM: price vs. return target" section
# for the exact numbers), not a guess: across 4 synthetic market regimes x 3
# seeds each (12 runs), return-mode beat price-mode on RMSE in 12/12 runs
# and on R2 in 12/12 runs, and never produced a negative R2 (price-mode did,
# twice). The likely cause: raw close prices are non-stationary, so a
# MinMaxScaler that (correctly, to stay leakage-safe) is fit only on the
# training period's price range can leave test-period prices scaled outside
# [0, 1] whenever the stock has trended since training -- the LSTM's
# saturating internal gates then have to extrapolate outside any range they
# were ever trained on. Daily returns don't drift the same way a price level
# does, so they don't hit this problem nearly as often.
#
# Directional accuracy was NOT meaningfully different between the two modes
# in that comparison (~48-50% either way, i.e. close to a coin flip) --
# switching to "return" mode is a genuine improvement in PRICE ERROR
# metrics, not a claim that it makes the model better at calling direction.
# "price" mode remains fully supported for anyone who wants it.
LSTM_PREDICTION_TARGET = os.getenv("LSTM_PREDICTION_TARGET", "return").strip().lower()
if LSTM_PREDICTION_TARGET not in ("return", "price"):
    raise ValueError("LSTM_PREDICTION_TARGET must be 'return' or 'price'.")

# Seed used to make runs reproducible (Python `random`, NumPy, TensorFlow).
# See reproducibility.py for exactly what this does and does not guarantee.
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# ---------------------------------------------------------------------------
# Data source selection
# ---------------------------------------------------------------------------
# "auto" tries yfinance first, then falls back to Upstox/Groww (if configured)
# for Indian stocks yfinance can't serve. Force one explicitly with
# "yfinance" / "upstox" / "groww" if you want to skip the fallback logic.
DATA_SOURCE = os.getenv("DATA_SOURCE", "auto")

# yfinance reliability: retry transient failures (rate limits, timeouts,
# connection resets) a few times with exponential backoff before giving up.
# A genuinely empty result (bad/delisted ticker) is never retried -- see
# data_sources/yfinance_source.py.
DATA_FETCH_MAX_RETRIES = int(os.getenv("DATA_FETCH_MAX_RETRIES", 3))
DATA_FETCH_RETRY_BACKOFF_SECONDS = float(os.getenv("DATA_FETCH_RETRY_BACKOFF_SECONDS", 1.5))
DATA_FETCH_TIMEOUT_SECONDS = float(os.getenv("DATA_FETCH_TIMEOUT_SECONDS", 15))

# --- Upstox (Indian NSE/BSE equities) ---
# From your app at https://account.upstox.com/developer/apps
UPSTOX_CLIENT_ID = os.getenv("UPSTOX_CLIENT_ID", "")
UPSTOX_CLIENT_SECRET = os.getenv("UPSTOX_CLIENT_SECRET", "")
UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "")

# --- Groww (Indian NSE/BSE equities) ---
# Option A (simplest, regenerate daily): https://groww.in/user/profile/trading-apis
GROWW_ACCESS_TOKEN = os.getenv("GROWW_ACCESS_TOKEN", "")
# Option B (automatable): https://groww.in/trade-api/api-keys
GROWW_API_KEY = os.getenv("GROWW_API_KEY", "")
GROWW_API_SECRET = os.getenv("GROWW_API_SECRET", "")

# ---------------------------------------------------------------------------
# LLM API keys (leave blank / unset to skip that provider)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")           # Grok
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Panel-level timeout ceiling (seconds) -- see ai_integrations/panel.py for
# why this exists alongside each provider's own ~30s HTTP timeout.
LLM_PANEL_TIMEOUT_SECONDS = float(os.getenv("LLM_PANEL_TIMEOUT_SECONDS", 45))

# Model names -- update these as providers release new versions
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
# gemini-2.0-flash was SHUT DOWN on 2026-06-01 (the cause of the old 404s). The default is a
# current free-tier-eligible model; availability is account-specific, so "Test connection"
# lists what your key can actually use instead of assuming.
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite")
# How the Gemini REST client authenticates: "header" (default, keeps the key
# out of the request URL) or "query" (legacy ?key=... escape hatch). See
# ai_integrations/gemini_client.py for why the default changed.
GEMINI_AUTH_MODE = os.getenv("GEMINI_AUTH_MODE", "header").strip().lower()
if GEMINI_AUTH_MODE not in ("header", "query"):
    raise ValueError("GEMINI_AUTH_MODE must be 'header' or 'query'.")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ---------------------------------------------------------------------------
# Ensemble weighting
# ---------------------------------------------------------------------------
# How much weight each source gets in the final blended prediction.
# The two quantitative models get most of the weight since only they see
# real numerical price history; LLMs contribute a smaller qualitative nudge
# based on reasoning over the trend summary they're given.
#
# Configurable via .env, consistent with every other setting in this file
# (previously these were hardcoded here despite the README describing them
# as configurable -- fixed for consistency).
_LR_WEIGHT = float(os.getenv("LR_WEIGHT", 0.5))
_LSTM_WEIGHT = float(os.getenv("LSTM_WEIGHT", 0.5))
# V17: 0 by default. No evidence an LLM panel adds out-of-sample value to a numeric price
# forecast; it is used for research commentary instead. Set >0 only if you have evaluated it.
_LLM_PANEL_WEIGHT = float(os.getenv("LLM_PANEL_WEIGHT", 0.0))

# V19: an LLM may influence the numerical price ONLY with an explicit acknowledgement that a documented, leakage-free,
# chronological out-of-sample experiment shows incremental value over the same forecast without it. Without it the weight
# is forced to 0 and the reason is exposed (LLM_WEIGHT_NOTE), never silently honoured.
LLM_EVIDENCE_ACK_VALUE = "I_HAVE_OUT_OF_SAMPLE_EVIDENCE"
LLM_WEIGHT_NOTE = None
if _LLM_PANEL_WEIGHT > 0 and os.getenv("LLM_PANEL_WEIGHT_EVIDENCE_ACK", "") != LLM_EVIDENCE_ACK_VALUE:
    LLM_WEIGHT_NOTE = (f"LLM_PANEL_WEIGHT={_LLM_PANEL_WEIGHT} was IGNORED (forced to 0): no out-of-sample evidence "
                       f"acknowledgement. See docs/V19_FINAL_AUDIT.md.")
    _LLM_PANEL_WEIGHT = 0.0
elif _LLM_PANEL_WEIGHT > 0:
    LLM_WEIGHT_NOTE = "LLM contributes to the numerical forecast by explicit operator acknowledgement; this app has NOT validated it."

if min(_LR_WEIGHT, _LSTM_WEIGHT, _LLM_PANEL_WEIGHT) < 0:
    raise ValueError("LR_WEIGHT, LSTM_WEIGHT, and LLM_PANEL_WEIGHT must all be non-negative.")
if _LR_WEIGHT + _LSTM_WEIGHT + _LLM_PANEL_WEIGHT <= 0:
    raise ValueError("LR_WEIGHT + LSTM_WEIGHT + LLM_PANEL_WEIGHT must sum to something positive.")

ENSEMBLE_WEIGHTS = {
    "linear_regression": _LR_WEIGHT,
    "lstm": _LSTM_WEIGHT,
    "llm_panel": _LLM_PANEL_WEIGHT,   # split evenly across however many LLMs respond
}

# Optional dynamic weighting: recompute LR/LSTM weights from walk-forward
# validation performance instead of using the fixed weights above. See
# ensemble.compute_dynamic_weights() for the method. "fixed" preserves the
# prior/default behavior exactly. The LLM panel's weight is never increased
# dynamically -- see compute_dynamic_weights() docstring for why.
ENSEMBLE_WEIGHTING_MODE = os.getenv("ENSEMBLE_WEIGHTING_MODE", "fixed").strip().lower()
if ENSEMBLE_WEIGHTING_MODE not in ("fixed", "dynamic"):
    raise ValueError("ENSEMBLE_WEIGHTING_MODE must be 'fixed' or 'dynamic'.")

# Absolute floor applied to each of LR/LSTM's weight when dynamic weighting
# is used, so a small/noisy validation sample can never push one model's
# weight to ~0% and let the other one dominate at ~100%.
ENSEMBLE_MIN_WEIGHT_FLOOR = float(os.getenv("ENSEMBLE_MIN_WEIGHT_FLOOR", 0.15))

WEBAPP_SECRET_KEY = os.getenv("WEBAPP_SECRET_KEY", "")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
# A loaded checkpoint older than this logs a clear "this may be stale"
# note (it's still used -- market data and conditions can meaningfully
# change over a week+ of trading days, but that's the user's call once
# they're told about it, not something to silently hide or block on).
CHECKPOINT_STALE_WARNING_DAYS = int(os.getenv("CHECKPOINT_STALE_WARNING_DAYS", 7))

# ---------------------------------------------------------------------------
# Trading / order execution settings
# ---------------------------------------------------------------------------
# SAFE BY DEFAULT: every run is a PAPER (simulated) trade unless you
# explicitly set ENABLE_LIVE_TRADING=true here AND pass --live on the CLI.
# Both gates must be true for any real order to reach a real broker.
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"

DEFAULT_BROKER = os.getenv("DEFAULT_BROKER", "paper")   # paper | upstox | groww

# Shared secret required by the webapp's live-trading routes (/trade in live
# mode, /trade/confirm, /trade/execute). Empty by default, which means the
# webapp REFUSES to place live orders at all -- see webapp/app.py's
# _live_trading_auth_state(). This is deliberately fail-closed: enabling
# ENABLE_LIVE_TRADING alone is not enough to make real orders reachable over
# HTTP; an operator must also set this. It is never rendered into a page or
# an error message, only compared against what the operator submits.
WEBAPP_LIVE_TRADING_TOKEN = os.getenv("WEBAPP_LIVE_TRADING_TOKEN", "")

# Minimum acceptable length for the above. Short shared secrets are
# brute-forceable over a local network in seconds, so a too-short value is
# rejected outright (live trading stays disabled) rather than quietly
# accepted as though it were protection.
WEBAPP_LIVE_TRADING_TOKEN_MIN_LENGTH = int(
    os.getenv("WEBAPP_LIVE_TRADING_TOKEN_MIN_LENGTH", 32)
)

# Signal thresholds: the ensemble's predicted next-day % change must clear
# these to generate a BUY or SELL signal at all (otherwise -> HOLD).
BUY_THRESHOLD_PCT = float(os.getenv("BUY_THRESHOLD_PCT", 0.5))
SELL_THRESHOLD_PCT = float(os.getenv("SELL_THRESHOLD_PCT", -0.5))

# Risk limits applied to every order regardless of broker (paper or live).
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", 0.10))     # max 10% of capital per trade
MAX_ORDER_VALUE = float(os.getenv("MAX_ORDER_VALUE", 10000))       # hard cap per single order
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", 5))

# Duplicate-order protection: refuse an identical LIVE order (same ticker,
# same side, same quantity) submitted within this many seconds of one that
# already went through. Guards against the classic double-submit -- a
# re-run script, a retried request, an impatient second click -- placing two
# real orders. 0 disables it. Paper orders are never blocked by this: they
# cost nothing and blocking them would make the simulator behave differently
# from the thing it is simulating.
DUPLICATE_ORDER_WINDOW_SECONDS = int(os.getenv("DUPLICATE_ORDER_WINDOW_SECONDS", 60))

# Paper broker settings
PAPER_STARTING_CASH = float(os.getenv("PAPER_STARTING_CASH", 100000))
PAPER_PORTFOLIO_PATH = os.getenv("PAPER_PORTFOLIO_PATH", ".paper_portfolio.json")

TRADE_LOG_PATH = os.getenv("TRADE_LOG_PATH", "outputs/trade_log.csv")

# ---------------------------------------------------------------------------
# Additional risk controls (Phase 6) -- ALL DISABLED BY DEFAULT (0 means "off").
# Existing behavior is unchanged unless a user explicitly sets these in .env.
# These are evaluated in trading/risk_manager.py and, where wired in,
# trading/trade_executor.py. They apply to the paper broker (which tracks
# an average entry price locally); live brokers would additionally need
# their own position/cost-basis lookup to enforce stop-loss/take-profit,
# which is broker-API-specific and out of scope for this pass (documented
# limitation in README).
# ---------------------------------------------------------------------------
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 0))              # e.g. 5 -> exit a position down 5% from entry
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", 0))          # e.g. 10 -> exit a position up 10% from entry
MAX_PORTFOLIO_EXPOSURE_PCT = float(os.getenv("MAX_PORTFOLIO_EXPOSURE_PCT", 0))  # e.g. 50 -> cap total invested value at 50% of portfolio
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", 0))  # e.g. 3 -> block new BUYs once today's realized loss hits 3%

# Minimum number of out-of-sample walk-forward folds required, for EACH of
# LR and LSTM, before dynamic ensemble weighting trusts their RMSE enough to
# compute real weights from it (see ensemble.compute_dynamic_weights).
# Below this, it transparently falls back to the fixed weights rather than
# basing weights on too small a sample.
#
# This is also used by pipeline.py to decide how many LSTM walk-forward
# folds to REQUEST when dynamic weighting is used without a full
# --walk-forward run -- it must never request fewer folds than this
# threshold requires, or dynamic weighting becomes structurally unreachable
# regardless of how much data is available (a real bug that existed here
# before: the "cheap" dynamic-weights-only path requested 2 LSTM folds while
# the safety check required 3, so it always fell back even with plenty of
# data). Keeping both sides driven by this single constant prevents that
# class of bug from reappearing.
DYNAMIC_WEIGHTS_MIN_FOLDS = int(os.getenv("DYNAMIC_WEIGHTS_MIN_FOLDS", 3))

# ---------------------------------------------------------------------------
# Backtesting (Phase 5)
# ---------------------------------------------------------------------------
BACKTEST_TRANSACTION_COST_PCT = float(os.getenv("BACKTEST_TRANSACTION_COST_PCT", 0.1))  # % of trade value, each way
BACKTEST_INITIAL_CAPITAL = float(os.getenv("BACKTEST_INITIAL_CAPITAL", 100000))
# Used only to compute the Sharpe ratio; 0 is a common simplifying default.
BACKTEST_RISK_FREE_RATE_ANNUAL_PCT = float(os.getenv("BACKTEST_RISK_FREE_RATE_ANNUAL_PCT", 0))
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# V17: market data freshness, AI providers, news, web app
# ---------------------------------------------------------------------------
def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)

# A quote is only "live" if the provider is known to supply real-time data AND its
# source timestamp is at most this old. Daily/EOD data can never be "live".
LIVE_MAX_AGE_SECONDS = _env_float("LIVE_MAX_AGE_SECONDS", 20)
QUOTE_TTL_SECONDS = _env_float("QUOTE_TTL_SECONDS", 15)       # cache TTL for a quote
HISTORY_TTL_SECONDS = _env_float("HISTORY_TTL_SECONDS", 900)  # cache TTL for daily history
HISTORY_FAILURE_TTL_SECONDS = _env_float("HISTORY_FAILURE_TTL_SECONDS", 30)  # a failed history fetch is not retried for this long
# A delayed quote older than this relative to its own expected delay is flagged stale.
DELAYED_STALE_AFTER_SECONDS = _env_float("DELAYED_STALE_AFTER_SECONDS", 1800)
PROVIDER_MAX_CONCURRENCY = int(_env_float("PROVIDER_MAX_CONCURRENCY", 4))
FORECAST_CACHE_TTL_SECONDS = _env_float("FORECAST_CACHE_TTL_SECONDS", 6 * 3600)

# AI providers (all optional). Model ids are account-specific -- see README.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "")
AI_REQUEST_TIMEOUT_SECONDS = _env_float("AI_REQUEST_TIMEOUT_SECONDS", 25)
AI_GLOBAL_BUDGET_SECONDS = _env_float("AI_GLOBAL_BUDGET_SECONDS", 40)
AI_FAILURE_COOLDOWN_SECONDS = _env_float("AI_FAILURE_COOLDOWN_SECONDS", 120)
AI_RESEARCH_CACHE_SECONDS = _env_float("AI_RESEARCH_CACHE_SECONDS", 1800)

# News: comma-separated "Name|URL" entries; "{symbol}" in a URL is replaced by the ticker.
# Empty by default -- the app never invents headlines.
NEWS_FEEDS = os.getenv("NEWS_FEEDS", "")
NEWS_TIMEOUT_SECONDS = _env_float("NEWS_TIMEOUT_SECONDS", 8)

# Web app: rate limit for expensive JSON endpoints (requests per minute per client IP).
WEBAPP_EXPENSIVE_RPM = int(_env_float("WEBAPP_EXPENSIVE_RPM", 12))

# V18: background jobs. The classic UI forms send async=1 and ALWAYS get an immediate 202 + job id. A client that does not
# send async (scripts, older integrations that expect the synchronous contract) waits at most
# LEGACY_PREDICT_WAIT_SECONDS for the job, then also gets 202. A running job is reported "timeout" after JOB_TIMEOUT_SECONDS.
LEGACY_PREDICT_WAIT_SECONDS = _env_float("LEGACY_PREDICT_WAIT_SECONDS", 120.0)
JOB_TIMEOUT_SECONDS = _env_float("JOB_TIMEOUT_SECONDS", 900)

# V19: drop a daily bar whose session is still in progress (never treat an intraday price as a settled close).
DROP_OPEN_SESSION_BAR = os.getenv("DROP_OPEN_SESSION_BAR", "true").lower() != "false"

# V19: warm TensorFlow/LSTM tracing in a background thread when the web server starts (the first forecast otherwise pays it).
WARMUP_ON_START = os.getenv("WARMUP_ON_START", "true").lower() != "false"
