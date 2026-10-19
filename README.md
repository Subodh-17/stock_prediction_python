# Multi-AI Stock Prediction & Trading Ensemble (Python)

Next-day stock closing price prediction that blends three independent
sources into one ensemble forecast -- and can act on that forecast by
placing buy/sell orders (simulated by default, real if you explicitly opt in).

1. **Linear Regression** -- interpretable baseline on lagged prices + simple
   technical features (moving averages, volatility, daily return).
2. **LSTM** -- a stacked recurrent network trained on 30-day sliding windows
   of price history, for nonlinear/longer-memory patterns the linear model
   can't capture.
3. **LLM Panel** -- Claude, ChatGPT, Gemini, Grok, and DeepSeek are each sent
   a plain-text summary of the recent price trend and asked for a direction,
   estimated % change, confidence, and one-line reasoning. Every response is
   validated before use (well-formed JSON, `direction` in {up, down, flat},
   confidence in [0, 1], and an estimated % change within a sane bound --
   anything else, including malformed JSON, missing fields, or a
   likely-hallucinated magnitude, is excluded from the panel rather than
   trusted). Valid responses are confidence-weighted and averaged into a
   "panel" estimate. One provider erroring, timing out, or returning garbage
   never crashes the run or the other providers' results.

The three sources are blended into one final predicted price -- either with
**fixed weights** (default 35% LR / 45% LSTM / 20% LLM panel, configurable),
or, optionally, **dynamic weights** derived from each numerical model's
walk-forward validation performance (see "Validation methodology" below).
That prediction can then feed a simple, transparent BUY/SELL/HOLD rule
(`trading/`), executed against a simulated paper broker by default, or a
real Upstox/Groww account if you explicitly enable it.

This project covers effectively any stock on any major exchange: a
multi-source data layer (`data_sources/`) tries **yfinance** first (US, most
Indian stocks, and most other world exchanges), and automatically falls back
to **Upstox** or **Groww** -- via your own trading account on either -- for
Indian NSE/BSE equities yfinance can't serve. See "Data sources" below.

**This is a standalone Python project.** The Scilab implementation lives in
its own separate project (`stock_prediction_scilab/`) -- it doesn't import
anything from here, and can be used entirely independently.

**Important:** this is an educational / research project, not investment
advice. LLMs have no special ability to predict markets, and neither do LR
or LSTM models fit purely on historical prices. Trading -- especially live,
real-money trading -- carries real risk of loss. See "Trading" below before
ever setting `ENABLE_LIVE_TRADING=true`.

> **This system is an experimental financial-analysis and trading
> prototype. Historical performance and model predictions do not guarantee
> future returns.**

Every prediction is also reported with an approximate confidence/model-
agreement summary (see "Prediction confidence & uncertainty" below) --
explicitly labeled as a heuristic, not a statistical guarantee.

> ### Read this first
>
> **Historical and backtest performance does NOT guarantee future returns.**
> Nothing in this project can predict the stock market. Linear Regression and
> the LSTM are fit to past prices; the LLM panel is qualitative opinion from
> language models. Every backtest in this repository is a **simulation** over
> a held-out historical period under explicitly stated assumptions -- it is
> not evidence that the strategy is profitable, and it is not a forecast.
>
> "Directional Accuracy" measures how often the predicted *direction* was
> right on held-out data. It is not proof the model can predict future
> prices, and a figure above 50% on one period says nothing about the next.
>
> This is an educational tool, not financial advice. Live trading is disabled
> by default and requires deliberate configuration; you are solely
> responsible for anything you enable.

> ## V20 notice -- read this first
> V20 audited V19 against its own code and fixed 5 confirmed defects, most importantly: **backtest trade-level P&L left out the entry
> fee** (win rate, profit factor and average trade return were overstated), **the classic `/predict` form accepted any text as a ticker**,
> and the data-quality checks missed frozen feeds, zero-volume stretches and timezone-aware indexes. Model comparisons now carry
> **confidence intervals and MASE**, and "beats naive" requires the 99% interval to exclude zero (the 95% interval proved anti-conservative
> in a calibration experiment). See [`docs/V20_ENGINEERING_AUDIT.md`](docs/V20_ENGINEERING_AUDIT.md) and
> [`docs/V20_RELEASE_NOTES.md`](docs/V20_RELEASE_NOTES.md). **Unchanged and still true: no market-data or AI provider has been verified
> against a live endpoint, and no real-market model-evaluation results exist.**
>
> ## V19 notice (still applies)
> V19 re-audited V18 against its own code and fixed 12 defects (full list with tests in [`docs/V19_FINAL_AUDIT.md`](docs/V19_FINAL_AUDIT.md)),
> most importantly: a stale quote could be re-labelled `unverified` after caching (and then traded in paper mode), checkpoints did not check
> library versions, unfinished sessions were still used by the classic flow, `/api/signal` and `/api/backtest` were synchronous, and the UI
> printed `[object HTMLSpanElement]` for cached quotes. **Still true: no market-data or AI provider has been verified against a live
> endpoint** (no network/credentials in the build environment), and no real-market model-evaluation results exist -- use
> `scripts/verify_providers.py` and `scripts/evaluate_universe.py` with your own access.
>
> ## V18 notice (still applies)
> V18 corrects methodological and compatibility problems found in V17 (prediction-interval calibration, checkpoint provenance,
> a Yahoo/TensorFlow import-order failure, the dependency lock, legacy exchange routing, background jobs, accessibility).
> **See [`docs/V18_CHANGES.md`](docs/V18_CHANGES.md)** for the verification matrix. In short: tests and a real-browser audit
> were run; **no market-data or AI provider was verified against a live endpoint** (the build environment had no network access
> to them and no credentials) -- run `python scripts/verify_providers.py` with your own keys to verify them.
>
> ## V17 notice (still applies)
> This version adds a quote/freshness layer, a background forecast service, a baseline-aware evaluation,
> a free-tier-first AI provider layer, and a new workstation UI at `/`. **See [`docs/V17_CHANGES.md`](docs/V17_CHANGES.md)**
> for what changed, what was verified (and what could not be), the methodology, and known limitations.
>
> Corrections to earlier statements in this README:
> * The "12 runs across 4 synthetic regimes" comparison of price vs. return targets (LSTM section) used **synthetic
>   data**. It is an engineering heuristic, not evidence about real markets.
> * `sample_outputs/` (charts from a synthetic-data run) was removed: it was not real market output.
> * The default Gemini model `gemini-2.0-flash` was shut down by Google on 2026-06-01 -- the cause of the old 404s.
> * LLMs no longer carry a fixed 20% weight in the numerical price (`LLM_PANEL_WEIGHT` now defaults to 0).
> * Test count history: 380 (V18), 442 (V19), 545 (V20); see "Running the tests".
> * Nothing in this build was verified against live Yahoo/Upstox/Groww/AI-provider endpoints (the development
>   sandbox could not reach them). Provider behaviour is covered by mocked tests plus opt-in live checks you run yourself.

## Project structure

```
stock_prediction_python/
├── main.py                        # Predict only -- prints metrics, saves charts/report
├── trade.py                       # Predict THEN act -- generates a signal and trades on it
├── pipeline.py                    # Shared LR + LSTM + LLM panel + ensemble logic (used by both)
├── config.py                      # All settings (ticker, hyperparams, API/data-source/trading keys)
├── .env.example                   # Copy to .env and fill in your keys (every setting documented)
├── requirements.txt               # Unpinned install path (portable across platforms)
├── requirements-lock.txt          # Pinned runtime set (verified: Python 3.12, Linux)
├── security.py                    # Central secret redaction + safe upstream-error handling
├── logging_utils.py               # Structured logging with redaction enforced at the logger
├── docs/API.md                    # JSON API contract: schemas, status codes, error shapes
├── data_loader.py                 # Feature engineering + LSTM windowing (source-agnostic)
├── data_sources/                  # Multi-source stock data layer
│   ├── base.py                    # Date-range helpers + exchange-suffix detection
│   ├── yfinance_source.py         # Default: US + most Indian + most world exchanges
│   ├── upstox_source.py           # Fallback: Indian NSE/BSE via your Upstox account
│   ├── groww_source.py            # Fallback: Indian NSE/BSE via your Groww account
│   └── __init__.py                # fetch_stock_data(...) -- auto-fallback resolver
├── ensemble.py                    # Blends LR + LSTM + LLM panel predictions (fixed or dynamic weights)
├── validation.py                  # Walk-forward (expanding-window) validation for LR + LSTM
├── backtest.py                    # Look-ahead-bias-free backtest vs. Buy & Hold
├── confidence.py                  # Approximate prediction confidence / model-agreement layer
├── checkpoints.py                 # Optional LSTM checkpointing (train once, reuse; ticker/config-checked)
├── reproducibility.py             # Seeds Python/NumPy/TensorFlow + captures run metadata
├── visualize.py                   # Chart generation
├── models/
│   ├── linear_regression_model.py
│   └── lstm_model.py
├── ai_integrations/
│   ├── base_client.py             # Shared prompt template + JSON parsing
│   ├── claude_client.py
│   ├── openai_client.py           # ChatGPT
│   ├── gemini_client.py
│   ├── grok_client.py
│   ├── deepseek_client.py
│   └── panel.py                   # Queries all 5 providers concurrently
├── trading/                       # Buy/sell decision + execution (see "Trading" below)
│   ├── signal_generator.py        # Ensemble prediction -> BUY/SELL/HOLD
│   ├── risk_manager.py            # Position sizing, daily trade limit, hard order caps
│   ├── trade_log.py               # Append-only audit log of every decision
│   ├── trade_executor.py          # Orchestrates signal -> risk checks -> confirm -> execute
│   └── brokers/
│       ├── base_broker.py         # Common interface every broker implements
│       ├── paper_broker.py        # Simulated, no real money -- the default
│       ├── upstox_broker.py       # Real orders via Upstox (requires opt-in)
│       └── groww_broker.py        # Real orders via Groww (requires opt-in)
├── webapp/                        # Browser front end for everything above (see "The website")
│   ├── app.py                     # Flask routes -- wraps pipeline.py + trading/, no logic duplicated
│   ├── templates/                 # Jinja2 HTML templates
│   ├── static/style.css           # Design system
│   ├── export_utils.py            # CSV / PDF report builders
│   └── test_app.py                # 49 functional tests via Flask's test client
├── tests/                         # See "Running the tests" below
```

## Architecture

```
                    ┌─────────────────────┐
                    │   data_sources/      │   yfinance -> Upstox -> Groww
                    │   (retry + fallback) │   (auto .NS/.BO resolution)
                    └──────────┬───────────┘
                               │ raw OHLCV (chronological)
                               ▼
                    ┌─────────────────────┐
                    │   data_loader.py     │   feature engineering,
                    │  (leakage-safe split)│   chronological train/test split,
                    └──────────┬───────────┘   scaler fit on TRAIN ONLY
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
  │ Linear Regression│ │      LSTM        │ │   ai_integrations/   │
  │  (models/)        │ │   (models/)       │ │  5-provider LLM panel │
  └────────┬─────────┘ └────────┬─────────┘ └──────────┬───────────┘
           │ price                │ price                │ direction/confidence
           │                      │                      │ (validated, invalid
           │                      │                      │  responses dropped)
           └──────────┬───────────┴──────────┬───────────┘
                       ▼                      ▼
              ┌─────────────────┐   ┌─────────────────────┐
              │   ensemble.py    │   │    validation.py      │
              │ (fixed or dynamic│◄──┤ (walk-forward folds,  │
              │  weights)         │   │  feeds dynamic weights)│
              └────────┬─────────┘   └─────────────────────┘
                       │ final predicted price
                       ▼
              ┌─────────────────┐        ┌─────────────────┐
              │trading/          │        │   backtest.py     │
              │signal_generator  │───────▶│ (simulated, vs.    │
              │-> BUY/SELL/HOLD  │        │  Buy & Hold)        │
              └────────┬─────────┘        └─────────────────┘
                       │
              ┌────────▼─────────┐
              │ risk_manager.py   │  position sizing, daily trade cap,
              │ (+ stop-loss/     │  daily loss limit, exposure cap
              │  take-profit)     │
              └────────┬─────────┘
                       ▼
              ┌─────────────────┐
              │ trade_executor.py │ -> paper broker (default) or
              │                    │    live broker (explicit opt-in)
              └─────────────────┘
```

Both `main.py` (predict only) and `trade.py` (predict + act) share the exact
same `pipeline.py` -> `ensemble.py` core; `webapp/app.py` wraps the same
pipeline again for the browser UI. No logic is duplicated between the three
entry points.

## Setup

**Supported Python: 3.12** (the only version this V18 lock was resolved and tested on). Platform of that verification:
**Linux x86_64, CPU only (`tensorflow-cpu`)**. Windows and macOS are **not verified for V18** -- see below.

| File | Use it when |
|---|---|
| `requirements.txt` + `webapp/requirements.txt` | You just want it to install (unpinned). |
| **`requirements-lock.txt`** | You want the exact, verified runtime set (Python 3.12, Linux). 80 pins. |
| `requirements-dev-lock.txt` | You also want to run the tests (adds pytest; includes the runtime lock). |
| `requirements-dev.txt` | Unpinned dev install. |
| `requirements-lock-windows-py311-legacy.txt` | The previous Windows/Python 3.11.0 lock (user-verified for V16 only; **not re-run against V18**). |

```powershell
# Windows PowerShell -- unpinned install on Python 3.12 (the verified path is the Linux lock; see docs/V18_CHANGES.md)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r webapp/requirements.txt
Copy-Item .env.example .env
```
```bash
# Linux, exactly the verified set
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt          # runtime only
pip install -r requirements-dev-lock.txt      # + pytest, to run the tests
cp .env.example .env
```

The lock was regenerated in V18 from a clean resolution, then installed into a *second* clean virtual environment, which
produced an identical 80-package set (`pip check`: no broken requirements). Two things that resolution revealed:
`anthropic` really does depend on `httpx2` (the old lock's note claiming it was not a real package was wrong), and a
**TensorFlow-before-yfinance import-order conflict** that breaks Yahoo requests (fixed by `_native_preload.py`; details in
`docs/V18_CHANGES.md`).

Tested versions (Python 3.12.3): numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, tensorflow-cpu 2.21.0, keras 3.15.1,
yfinance 1.7.0, curl_cffi 0.16.3, matplotlib 3.11.2, anthropic 1.13.0, openai 3.28.0, requests 2.34.2, python-dotenv 1.2.4,
flask 3.1.3, reportlab 5.0.1, pytest 9.1.1.

## Data sources -- covering "all types of stocks"

`--source auto` (the default) tries sources in this order:

1. **yfinance** -- no setup needed. Covers:
   - **US exchanges** -- no suffix (`AAPL`, `MSFT`, `TSLA`)
   - **India** -- `.NS` (NSE) or `.BO` (BSE), e.g. `RELIANCE.NS`, `TCS.BO`
   - **Most other major world exchanges**, via their own suffix:

     | Exchange | Suffix | Example |
     |---|---|---|
     | London (LSE) | `.L` | `VOD.L` |
     | Tokyo (TSE) | `.T` | `7203.T` |
     | Hong Kong (HKEX) | `.HK` | `0700.HK` |
     | Shanghai / Shenzhen | `.SS` / `.SZ` | `600519.SS` |
     | Toronto (TSX) | `.TO` | `SHOP.TO` |
     | Australia (ASX) | `.AX` | `BHP.AX` |
     | Germany (Xetra) | `.DE` | `SAP.DE` |
     | Paris (Euronext) | `.PA` | `MC.PA` |
     | Singapore (SGX) | `.SI` | `D05.SI` |
     | South Korea (KRX) | `.KS` / `.KQ` | `005930.KS` |
     | Brazil (B3) | `.SA` | `PETR4.SA` |

     (Not exhaustive -- yfinance/Yahoo Finance's own ticker lookup is the
     authoritative source if a suffix isn't listed here.)

2. **Upstox** -- if yfinance comes up empty or rate-limited for an Indian
   ticker, and you've set `UPSTOX_CLIENT_ID` / `_SECRET` / `_REDIRECT_URI`
   in `.env` (from your app at https://account.upstox.com/developer/apps).
   First run each day, it prints a login URL, you log in in a browser, and
   paste the code back -- that's Upstox's actual security model (tied to a
   real trading account), not something a script can skip. The token is
   cached and reused for the rest of the day.
3. **Groww** -- same purpose as Upstox, if you've set `GROWW_ACCESS_TOKEN`
   (simplest, regenerate daily from https://groww.in/user/profile/trading-apis)
   or `GROWW_API_KEY` / `_SECRET` (more automatable, from
   https://groww.in/trade-api/api-keys) in `.env`.

**Upstox and Groww are only ever tried for plausibly-Indian tickers.** A
ticker with a suffix that clearly targets a different exchange (e.g.
`VOD.L`) will never be sent to them, even if both are fully configured --
neither broker covers those markets, so trying would just waste a request.
A bare symbol with no suffix (`RELIANCE`) is treated as ambiguous: yfinance
itself is now retried with `.NS` then `.BO` appended (same symbol, just the
correct exchange suffix -- never a different security) before falling
through to Upstox/Groww, so most Indian-stock bare symbols resolve without
needing any broker credentials at all.

**Reliability.** yfinance requests retry transient failures (rate limits,
timeouts, connection resets) up to `DATA_FETCH_MAX_RETRIES` times (default
3) with exponential backoff, and use an explicit
`DATA_FETCH_TIMEOUT_SECONDS` request timeout (default 15s) so a hung
connection can't block a run indefinitely. A genuinely empty result (bad or
delisted ticker) is *not* retried -- that's a real answer, not a transient
glitch, and retrying it would just waste time before producing the same
correct error.

You can also force one source explicitly:
```bash
python main.py --ticker RELIANCE --source upstox
python main.py --ticker TCS --source groww
python main.py --ticker AAPL --source yfinance
```

Most users never need to touch Upstox or Groww at all -- they exist purely
as fallback coverage for Indian stocks that yfinance handles imperfectly.

## Running it (prediction only)

```bash
python main.py --ticker AAPL --period 2y
```

Other useful flags:
```bash
python main.py --ticker TSLA --epochs 40      # more LSTM training epochs
python main.py --ticker MSFT --skip-llm       # LR + LSTM only, no API calls/cost
python main.py --ticker RELIANCE.NS           # Indian stock via yfinance
python main.py --ticker VOD.L                 # London Stock Exchange via yfinance
python main.py --ticker AAPL --walk-forward   # + expanding-window validation (see below)
python main.py --ticker AAPL --dynamic-weights # ensemble weights from walk-forward performance
python main.py --ticker AAPL --backtest       # + simulated backtest vs. Buy & Hold
python main.py --ticker AAPL --dynamic-weights --backtest  # backtest replays the SAME dynamic
                                               # weighting methodology as the live prediction --
                                               # see "Backtest weighting" below
python main.py --ticker AAPL --checkpoint aapl_lstm         # train once, reuse the LSTM next time
python main.py --ticker AAPL --checkpoint aapl_lstm --retrain-checkpoint  # force a fresh retrain
python main.py --list-checkpoints             # list saved checkpoints and exit
```

This prints progress + metrics to the console: RMSE, MAE, R², MAPE (PRICE
ERROR metrics -- how close the predicted dollar figure was), and separately,
**Directional Accuracy** (the % of test-set days the model correctly called
the price direction, up or down -- unrelated to price error; a model can be
good at one and mediocre at the other). Directional accuracy is never
reported as "model accuracy" on its own. It writes to `outputs/`:
- `model_comparison.png` -- actual vs. LR vs. LSTM predicted price on the test set
- `llm_panel_votes.png` -- each LLM's estimated % change and confidence
- `final_ensemble.png` -- all component predictions vs. the final blended price
- `report.json` -- everything above in structured form (metrics, predictions,
  full LLM reasoning text, and walk-forward/backtest results if requested)

## Validation methodology

Every run evaluates both models on a single chronological train/test split
(never shuffled -- shuffling would leak future prices into training). This
is fast, but it's one snapshot of one held-out period, which can be lucky or
unlucky.

`--walk-forward` additionally runs **expanding-window walk-forward
validation** (`validation.py`): the model is retrained across several
sequential folds, each time training on everything up to that point and
validating on the next chronological chunk -- never shuffled, never using
future data to evaluate the past. This gives a distribution of RMSE / MAE /
R² / directional accuracy across folds rather than a single number, which is
a more honest picture of how the model would have performed if you'd
actually been retraining and using it over time. It costs extra runtime
(the LSTM gets retrained several more times, with fewer epochs than the main
model by default to keep this practical -- see `validation.py`'s docstring
for the exact tradeoff).

## LSTM: price vs. return target, and a "next day" prediction bug

Two real issues were found and fixed while reviewing the LSTM:

**1. Predicting price vs. predicting return.** The LSTM originally predicted
the next day's raw (MinMax-scaled) closing price. Because the scaler is
(correctly, to stay leakage-safe) fit only on the training period's price
range, a stock that trended since training leaves test-period prices scaled
outside `[0, 1]` -- and the LSTM's saturating internal gates handle that
extrapolation poorly. An actual empirical comparison (not a guess) across 4
synthetic market regimes (uptrend/low-vol, uptrend/high-vol, downtrend,
choppy) x 3 seeds each (12 runs total) found predicting the next day's
**return** (% change) instead beat predicting raw price on RMSE in **12/12**
runs and on R² in **12/12** runs (mean R² 0.93 vs. -1.21 across the 12
runs), and never produced a negative R² (price-mode did, twice). Returns are
more stationary than a price level, so they don't hit the same
scaled-out-of-range problem. `LSTM_PREDICTION_TARGET=return` is now the
default (`LSTM_PREDICTION_TARGET=price` remains fully supported). Honest
caveat: this did **not** meaningfully change directional accuracy in that
same comparison (~48-50% either way, close to a coin flip either mode) --
it's an improvement in price-error metrics specifically, not a claim that
the model got better at calling direction.

**2. The "next day" prediction was blind to the actual most recent day.**
`predict_next()` used to be called with the last row of the model's own
test set -- but because feature engineering shifts the prediction target by
one day and drops the resulting row with no known target (correctly, since
you can't have a training/test example whose label doesn't exist yet), that
"last row" is actually built from **yesterday's** data, predicting a day
that's already in the dataset and already known. Confirmed directly:
shocking the final row's Close by +500 changed the reported "next day
prediction" not at all, for both LR and LSTM. Fixed with
`data_loader.build_latest_feature_row()` / `build_latest_lstm_window()`,
which build a fresh row/window from the actual most recent day -- verified
the same shock test now correctly changes the prediction.

## Prediction confidence & uncertainty

Every prediction now also reports a `confidence.py`-generated summary:

```
Prediction:        $217.06
Expected change:   -0.34%
Model agreement:   High
Confidence:        Moderate
Approx. price range: $213.81 - $220.31
```

**This is explicitly an approximate heuristic, not a formally valid
statistical confidence interval.** Building one properly would require
modeling the actual residual distribution (quantile regression, conformal
prediction, or a Bayesian model) and validating its coverage -- none of
which this project's models do. What it actually computes:
- **Model agreement**: do LR/LSTM/(LLM panel) point the same direction, and
  how close are their predicted % changes to each other? ("High" = same
  direction and within 1 point of each other; "Low" = fewer than 2/3 agree
  on direction, or the spread exceeds 3 points -- documented heuristic
  thresholds, not derived from a statistical test.)
- **Approximate price range**: the final prediction +/- a blend of each
  model's own historical RMSE (walk-forward mean where available, else the
  single train/test split), weighted the same way the ensemble weighted
  those models for its point prediction.
- **Confidence label**: "Low" whenever model agreement is "Low" (the models
  disagreeing on direction should never be papered over by error-band math),
  or whenever the approximate error band is wider than the predicted move
  itself; otherwise "Moderate" or "Moderate-High".

## JSON API

Full contract -- request/response schemas, every status code and the error
shape -- is in [`docs/API.md`](docs/API.md). Summary:

```
GET  /api/health                                  -> {"status": "ok", ...}
POST /api/predict   {"ticker": "AAPL", ...}        -> full prediction + confidence + signal
POST /api/signal     {"ticker": "AAPL"}             -> just the BUY/SELL/HOLD call (fast path)
POST /api/backtest   {"ticker": "AAPL", "model": "ensemble"}  -> backtest metrics
```

All accept either a JSON body or form-encoded data. Every input is
validated (ticker format, `period` against a fixed allow-list, `epochs`
bounded to 1-500, `source`/`model` against allow-lists) and every expected
failure mode returns a structured JSON error with an appropriate status
code -- `400` for bad input, `422` for insufficient data, `502` for a data-
source outage, `500` (with a deliberately generic message, never the raw
exception text) for anything unexpected. `422` also covers
`all_models_failed` -- the case where no component model produced a usable
prediction at all.

**Metric naming in responses.** Each model's `metrics` block reports the
directional metric as **`directional_accuracy_pct`**. The older key
`accuracy_pct` is still present alongside it, unchanged, so existing
consumers and previously-saved `report.json` files keep working -- but the
bare word "accuracy" is genuinely misleading here (it measures whether the
predicted *direction* was right, not classification accuracy and not how
close the predicted price was), so new consumers should read
`directional_accuracy_pct`. Try it:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL", "epochs": 15, "skip_llm": true}'
```

## Model checkpointing

The LSTM can be trained once and reused instead of retraining on every run:

```bash
python main.py --ticker AAPL --checkpoint aapl_lstm          # trains + saves the first time
python main.py --ticker AAPL --checkpoint aapl_lstm          # loads it, skips training
python main.py --ticker AAPL --checkpoint aapl_lstm --retrain-checkpoint  # forces a fresh retrain
python main.py --list-checkpoints                             # see what's saved
```

A checkpoint (`checkpoints/<name>/`) stores the Keras model, its fitted
scaler, and metadata (ticker, lookback, prediction target, training row
count, timestamp, package versions). **Loading never silently reuses an
incompatible checkpoint**: if the saved ticker, lookback window, or
prediction target (return vs. price) doesn't match the current run, loading
is refused outright with a clear explanation and the model trains fresh
instead (then overwrites the checkpoint) -- a mismatch on any of these
would either crash on a shape mismatch or, worse, silently produce
predictions for the wrong stock. The loaded model is still evaluated
against the CURRENT run's own held-out test split (never against stale
cached numbers), so its reported RMSE/accuracy always reflects genuine
current-data performance.

**Staleness is flagged, not hidden.** A loaded checkpoint older than
`CHECKPOINT_STALE_WARNING_DAYS` (default 7) prints a clear note that market
conditions/data since it was saved aren't reflected in the loaded weights.
This is a warning, not a block -- an old checkpoint is still usable (it's
the user's call whether that's fine for their purpose), but it's never used
silently without saying how old it is.

### Checkpoint compatibility checks

A checkpoint is refused (never silently reused) when its saved metadata
disagrees with the current run on **ticker**, **lookback window**,
**prediction target** (return vs. price), or **train/test split**. The split
check closes a real leakage hole: the pipeline rebuilds test sequences at the
*current* `TRAIN_TEST_SPLIT`, but the checkpoint's weights and scaler came
from whatever split was in force when it was saved. Saved at `0.8` and re-run
at `0.6`, the rebuilt "test" window starts inside rows the model actually
trained on, and `lstm_eval` would report in-sample performance as held-out.
Checkpoints saved before this check existed don't record enough to be
verified, so they are refused too. Every refusal names the specific mismatch
and falls back to training fresh.

## Ensemble methodology

**Fixed weights** (default): `ENSEMBLE_WEIGHTS` in `.env`
(`LR_WEIGHT`=0.35, `LSTM_WEIGHT`=0.45, LLM panel=0.20 split evenly across
however many providers respond).

**Dynamic weights** (`--dynamic-weights` or `ENSEMBLE_WEIGHTING_MODE=dynamic`
in `.env`): runs walk-forward validation internally and re-splits the
LR/LSTM portion of the weight budget by inverse RMSE -- the model that was
more accurate across several out-of-sample folds gets more weight. An
absolute floor (`ENSEMBLE_MIN_WEIGHT_FLOOR`, default 0.15) keeps a small or
noisy validation sample from pushing either model to near-zero weight, and
the floor is applied so it always holds even after normalization (an
earlier version applied it before a final renormalization step, which could
in some cases push the "floored" weight back below the floor -- fixed).
If there isn't enough walk-forward data (fewer than `DYNAMIC_WEIGHTS_MIN_FOLDS`
folds for either model, default 3, or a non-finite RMSE), it transparently
falls back to the fixed weights rather than producing degenerate output.
Console/log output distinguishes the two outcomes explicitly:

```
Weighting mode: dynamic
LR validation RMSE:  1.8234 (5 folds)
LSTM validation RMSE: 4.1102 (3 folds)
LLM contribution (fixed, not derived from validation): 20%
Final weights -> LR: 65%  LSTM: 15%  LLM: 20%
```
or, if there isn't enough evidence:
```
Weighting mode: fixed_fallback
Reason: Insufficient walk-forward validation folds (need >= 3 for both LR
and LSTM) to compute reliable dynamic weights -- using configured fixed weights.
```

**The LLM panel's weight is never increased dynamically** -- it stays at
its configured share regardless of how confident any individual LLM
response sounds, so LLM output can never come to dominate the numerical
models.

**A real bug was found and fixed here**: `pipeline.py` used to request only
2 LSTM walk-forward folds for the "dynamic weights without a full
`--walk-forward` run" path, while the fallback safety check required at
least 3 folds before trusting the result -- an internal contradiction that
made `--dynamic-weights` (used alone) fall back to fixed weights on *every*
run, regardless of how much data was available, always reporting
"insufficient folds" even when that wasn't really true. Both numbers are
now driven by one `DYNAMIC_WEIGHTS_MIN_FOLDS` config constant so they can't
drift apart again; the fallback safety check itself was never weakened --
genuine data insufficiency (verified with a short synthetic series) still
correctly falls back.

**A second real bug, found in a later review**: `weights_used` (returned by
every prediction, and read by the backtest's weighting-mode auto-detection
below) used to always report the *pre-redistribution* weights when no LLM
panel was available -- so a `--skip-llm` run with `LR_WEIGHT`=0.35/
`LSTM_WEIGHT`=0.45/LLM=0.20 configured would still claim `llm_panel: 0.20`
in its output, even though the LLM contributed exactly 0% and that 20% had
actually been redistributed to LR (43.75%) and LSTM (56.25%). Fixed:
`weights_used` now always reflects the weights that were genuinely applied.

## Date alignment between the LR and LSTM series

`add_technical_features()` builds each row's features from data up to and
including day *d*, but its label is `Close.shift(-1)` -- day *d+1*'s close.
The LR test set therefore used to be reported against **feature** dates,
one trading day earlier than the day each target was actually realized,
while the LSTM series was already labeled by realization date.

That offset was invisible inside the LR model itself (features and labels
stayed correctly paired) but mattered everywhere the two series were lined
up against each other:

- the **LR+LSTM ensemble backtest** joins the two models on date, so it was
  blending an LR forecast of `close(d+1)` with an LSTM forecast of
  `close(d)` and scoring the result against `close(d+1)` -- the LSTM
  component was effectively contributing a forecast of a price already known
  at signal time, which damps the blended signal;
- the **CSV/PDF export table** put an `actual` of `close(d+1)` on the same
  row as an `lstm_predicted` for `close(d)`;
- the **actual-vs-predicted chart** drew the LR lines one day to the left;
- **walk-forward fold boundaries** were reported a day early, which made the
  backtest's "only folds that concluded strictly *before* this day" filter
  one day too permissive -- a fold that had in fact already observed day
  *d*'s close was admitted when weighting day *d*. That was a genuine, if
  small, look-ahead.

`data_loader.lr_target_dates()` now maps every feature row to the date its
target was realized, and both `prepare_linear_regression_data()` and
`validation.walk_forward_validate_lr()` label by that. The invariant is
tested directly: on every overlapping date, the LR and LSTM series must
agree on what the actual close was (`tests/test_final_pass.py`).

## When a model fails

A component whose prediction is missing, `NaN`, infinite, or non-positive is
**excluded** from the blend and its weight redistributed proportionally
across the survivors -- previously a single `NaN` from either model made the
entire ensemble `NaN`, so one misbehaving model took down a prediction the
other could still support. Exclusions are never hidden: each one is listed
in the result's `excluded_components`, and the excluded model's reported
weight is `0.0`. If *no* component is usable, `ensemble.AllModelsFailedError`
is raised rather than emitting a meaningless number that would flow onward
into the signal generator, the web UI and `report.json` looking real.

## Backtest weighting: consistency between prediction and backtest

A live prediction can use fixed or dynamic ensemble weights; a real
inconsistency existed here until this review -- the "ensemble" backtest
always used `config.ENSEMBLE_WEIGHTS` (fixed) regardless of what the live
prediction actually used, so `--dynamic-weights --backtest` would silently
show a fixed-weight backtest right next to a dynamic-weight prediction.

**Fixed, with genuine (not faked) historical dynamic weighting where
possible.** `backtest_from_pipeline_result(..., weighting_mode="auto")`
(the default) now matches whatever the live prediction actually used:

- **Live run was fixed** (or fell back to fixed): the backtest applies that
  same fixed split uniformly across the whole period. This is *not* a
  leakage risk -- a static config constant isn't derived from any data, so
  applying it uniformly to history doesn't use anything a strategy
  "couldn't have known" at any point in time.
- **Live run was dynamic**: the backtest recomputes a genuinely
  historically-valid weight split for *each backtest day*, using ONLY
  walk-forward folds whose test period concluded **strictly before** that
  day (`backtest._compute_historical_dynamic_weights`). This deliberately
  does **not** take today's final dynamic weights and apply them across the
  whole backtest -- that would let an early backtest day be influenced by
  validation folds that hadn't happened yet as of that day, exactly the
  future-information leak a backtest exists to avoid. Days without enough
  *prior* fold history yet (typically the earliest part of the backtest
  period) fall back to the fixed split for that day specifically,
  transparently -- reported as e.g. "Dynamic (84/98 days)" in the CLI/UI.

A related fix was needed to make this actually useful: the LSTM
walk-forward fold count for `--dynamic-weights` (without `--walk-forward`)
used to request *exactly* the safety-floor number of folds, which meant the
last fold's own test period always extended to near the end of the
dataset -- leaving **zero** folds old enough to precede a later backtest's
test window (confirmed empirically: 0 dynamic-weighted days out of ~100
before this fix). `--backtest` now signals this intent (`for_backtest=True`
internally) so a few extra folds are requested, giving real "runway" for
day-by-day dynamic weighting to actually engage.

`weighting_mode` can also be forced explicitly (`"fixed"` or `"dynamic"`)
via the API's `/api/backtest` `weighting_mode` field; forcing `"dynamic"`
without walk-forward fold data available raises a clear error rather than
silently faking it. The methodology actually used is always stated
explicitly -- CLI output, the webapp's backtest table ("Weighting" column),
`BacktestResult.summary_text()`, and `BacktestResult.to_dict()`
(`weighting_mode` / `weighting_detail` fields) all report it, never leaving
it implicit.

## Backtesting methodology

`--backtest` runs `backtest.py` -- a simple, transparent simulation of the
exact same BUY/SELL/HOLD threshold rule used live, applied to each model's
**held-out test-period predictions** (never anything the model was trained
on), for LR, LSTM, and an LR+LSTM blend, each compared against a Buy & Hold
baseline over the identical period.

Reports are split into two clearly distinct categories, never conflated:

**MODEL PREDICTION PERFORMANCE** (how close the forecast price was) --
RMSE, MAE, MAPE, R², directional accuracy -- reported separately per model,
*before* any backtest even runs (see the CLI output / dashboard).

**TRADING STRATEGY PERFORMANCE** (what a simulated trader following that
forecast would have made) -- total return, annualized return, max drawdown,
Sharpe ratio, **Sortino ratio** (like Sharpe but only penalizes downside
volatility -- undefined/`None` rather than fabricated when there are no
losing days to measure), **Calmar ratio** (annualized return / max
drawdown -- undefined when there's no drawdown to divide by), number of
trades, win rate, **profit factor** (total winning $ / total losing $ --
undefined, not infinity, when there are no losing trades), **average trade
return** (% return per closed round trip, position-size-independent), and
total transaction costs.

**Explicit timing assumptions** (also documented in `backtest.py`'s module
docstring -- read it before trusting the numbers):
- Day *i*'s signal compares a prediction for day *i*'s close (built using
  only data through day *i-1*) against day *i-1*'s actual close -- the same
  rule and thresholds as `trading/signal_generator.py`.
- A triggered trade is assumed to fill at day *i*'s own close price (a
  standard simplifying assumption for daily backtests -- no intraday or
  next-day-open modeling).
- Long-only, matching the live trading model (no shorting).
- Transaction costs (`BACKTEST_TRANSACTION_COST_PCT`, default 0.1%) are
  applied each way; no additional slippage model.

This is a fresh addition -- there was no backtesting engine in this project
before. **It is a simulation over historical held-out data, not a
prediction or guarantee about future performance.**

## Risk controls

All optional, all **disabled by default** (0 = off) so existing behavior is
unchanged unless you opt in via `.env`:

| Setting | Default | What it does |
|---|---|---|
| `MAX_POSITION_PCT` | 0.10 | Cap a single BUY at this fraction of available capital (always on) |
| `MAX_ORDER_VALUE` | — | Hard cap on a single order's dollar value (always on) |
| `MAX_TRADES_PER_DAY` | 5 | Blocks new trades once today's count is hit (always on) |
| `STOP_LOSS_PCT` | 0 (off) | Forces a SELL if a held position is down this % from its entry price |
| `TAKE_PROFIT_PCT` | 0 (off) | Forces a SELL if a held position is up this % from its entry price |
| `MAX_PORTFOLIO_EXPOSURE_PCT` | 0 (off) | Caps total position value (existing + new) as a % of portfolio |
| `DAILY_LOSS_LIMIT_PCT` | 0 (off) | Blocks new BUYs (never SELLs) once today's mark-to-market loss hits this % |

Stop-loss/take-profit and the daily loss limit are currently enforced for
**the paper broker only** -- it's the only broker this project tracks a
running average entry price and day-start portfolio value for locally. A
live broker would need its own cost-basis lookup via that broker's position
API to support the same checks; that's a documented limitation, not
something silently skipped.

Paper trading/simulation is always safe by construction (`.paper_portfolio.json`,
no real money). Live trading requires the full opt-in chain described below
-- it is never automatic.

## Trading -- turning a prediction into a buy/sell decision

```bash
python trade.py --ticker AAPL
```

This runs the exact same prediction pipeline as `main.py`, then:
1. Converts the ensemble's predicted % change into a **BUY / SELL / HOLD**
   signal via a simple, transparent threshold rule (`BUY_THRESHOLD_PCT` /
   `SELL_THRESHOLD_PCT` in `.env`, default ±0.5%). Anything in between is HOLD.
2. If it's a BUY or SELL, sizes the order (`risk_manager.py`: at most
   `MAX_POSITION_PCT` of available capital, or `MAX_ORDER_VALUE`, whichever
   is smaller -- both configurable in `.env`), and checks the daily trade
   count against `MAX_TRADES_PER_DAY`.
3. Executes it -- **against a simulated paper broker by default.** Every
   decision (including HOLDs and blocked/rejected attempts) is appended to
   `outputs/trade_log.csv` for a full audit trail.

### Safety model -- please read before using `--live`

**Every run is paper (simulated) trading unless ALL of the following are true:**

1. You pass `--mode live` on the command line, **and**
2. You have set `ENABLE_LIVE_TRADING=true` in your `.env` file by hand --
   this is deliberately *not* something any CLI flag can set, so a typo or a
   copy-pasted command can't silently turn on real trading, **and**
3. You pass `--broker upstox` or `--broker groww` (there's no default live
   broker), **and**
4. You type the exact confirmation phrase (`CONFIRM LIVE TRADE`) when
   prompted for that specific order -- unless you pass `--auto-confirm`,
   which is meant only for unattended automation you've already reviewed
   and accepted the risk of.

**Through the web UI there is a fifth, independent requirement:** you must
also set `WEBAPP_LIVE_TRADING_TOKEN` in `.env` to a random value of at least
32 characters, and type it into the live-trade form. This is **fail-closed**
-- with no token configured (the default), the live-trading routes refuse
outright with a 403, even when `ENABLE_LIVE_TRADING=true`. Reaching the page
is therefore never sufficient to place a real order; two separate, deliberate
edits to `.env` are needed first, and no HTTP request can perform either.

```bash
# generate one:
python -c "import secrets; print(secrets.token_hex(32))"
```

What this protection is and is not: it is a single shared operator secret,
compared in constant time, never rendered into a page or an error message,
and never stored in the session (only the fact that it was verified). It is
**not** a user-account system -- there are no per-user logins, roles, or
audit-by-identity. That was a deliberate trade-off for a single-user local
tool: real accounts would mean a user store, password hashing and a login
flow, which is a large change for a tool that binds to `127.0.0.1` by
default. If you expose this app to anything wider than your own machine, put
it behind a real authenticating reverse proxy; the shared secret alone is not
designed to be an internet-facing authentication layer.

```bash
# Paper trading (safe, default) -- simulated portfolio in .paper_portfolio.json
python trade.py --ticker RELIANCE --source upstox

# Real trading (only after ENABLE_LIVE_TRADING=true is set in .env)
python trade.py --ticker RELIANCE --source upstox --mode live --broker upstox
```

Other trading flags:
```bash
python trade.py --ticker AAPL --skip-llm                  # faster signal, LR+LSTM only
python trade.py --ticker AAPL --capital 5000               # override available capital
python trade.py --ticker AAPL --quantity 10                # trade an exact share count
                                                              # (still validated by risk_manager)
```

**This tool is not investment advice and does not know anything the models
don't.** A BUY signal is a mechanical output of a threshold rule applied to
statistical model predictions -- not a recommendation from Claude or anyone
else. You are solely responsible for any trade this tool executes on your
behalf, especially in `--live` mode.

## The website

Everything above also runs as a real browser-based website, not just a CLI
-- same pipeline, same trading module, same safety gates, just with a UI.

```bash
pip install -r webapp/requirements.txt
python webapp/app.py
```

Then open **http://127.0.0.1:5000**. The homepage is a prediction form (with
optional walk-forward validation, dynamic ensemble weighting, and backtest
checkboxes -- all add runtime); the results page shows an "Ensemble Vote"
strip (each model's individual direction converging into one final signal),
the ensemble weights actually used (and whether they were fixed or
dynamic), a confidence/model-agreement summary, price error and
directional accuracy metrics (RMSE, MAE, R², MAPE, and Directional Accuracy,
clearly labeled as a separate metric from price error, for both LR and
LSTM), walk-forward validation and backtest results when requested (the
backtest table states its own weighting methodology explicitly -- see
"Backtest weighting" above), and the
charts (each with a legend and "Time" / "Stock Price" axis labels), with a
**Paper Trade** button and, if `ENABLE_LIVE_TRADING=true` is set, a **Live
Trade** button. **Export CSV** and **Export PDF** buttons download the full
results -- metrics, signal, and the date-aligned actual/predicted table
(CSV), or a formatted one-page report with the charts embedded (PDF). No API
keys are ever sent to or exposed in frontend code -- every LLM/broker call
happens server-side in `pipeline.py`/`trading/`.

**Live trading through the website has its own safety design**, adapted
from the CLI's interactive confirmation (which can't work in a web request
-- there's no terminal attached to an HTTP call). Instead: clicking Live
Trade goes to a dedicated confirmation page showing the exact order, which
requires typing the same confirmation phrase the CLI uses (`CONFIRM LIVE
TRADE`) into a form field, plus a one-time token minted specifically for
that page load. Both are checked on the server before anything reaches a
broker -- confirmed by dedicated tests that a wrong phrase or a
forged/reused token never reaches the trade-execution function, not just
that the page displays correctly.

**Security notes if you plan to actually use this:**
- Binds to `127.0.0.1` (localhost only) by default. This app has no
  authentication layer, and it can place real orders when
  `ENABLE_LIVE_TRADING=true` -- do not expose it on a network without adding
  auth first.
- Flask's debug mode (the interactive in-browser debugger) is off by
  default and should stay off in any environment where live trading could
  be enabled; that debugger is a known code-execution risk if it's ever
  reachable by anyone but you.
- Results are cached in server memory (not a database) keyed to your
  browser session -- fine for the single-user local tool this is designed
  to be, restarting the server clears everything.

Test it (49 checks, no server or browser needed -- Flask's test client
sends real requests in-process):
```bash
python webapp/test_app.py
```

## Running the tests

No API keys, broker credentials, or internet required -- every test uses
synthetic data and/or mocked network calls, so you can verify the whole
project works before ever touching real credentials:

```bash
python tests/test_pipeline_synthetic.py         # LR + LSTM + ensemble + charts, synthetic data
python tests/test_metrics_utils.py              # directional accuracy metric, hand-verified
python tests/test_llm_integration.py            # LLM JSON parsing + panel blending, mocked
python tests/test_data_sources.py               # yfinance/Upstox/Groww data layer, mocked
python tests/test_trading.py                    # signal, risk limits, paper broker, safety gates
python tests/test_brokers.py                    # Upstox/Groww order placement, mocked
python tests/test_integration_pipeline_trade.py # full predict -> trade flow, end to end
python webapp/test_app.py                       # website routes + live-trade safety gates, end to end

# Proper pytest suite (from an actual pytest run -- see "Running
# the tests" below for the exact command) covering the leakage fix, ticker propagation,
# the dynamic-weighting fold-count bug, the LSTM return-vs-price comparison,
# the "next day prediction" bug, walk-forward validation, LLM output validation
# (including the Gemini API-key-redaction security fix), data-source retries,
# the backtest engine (incl. Sortino/Calmar/profit factor), risk controls,
# data quality validation, the confidence/uncertainty layer, the JSON API,
# and model checkpointing:
pytest tests/test_phase_improvements.py -v

# Regression tests from the final production-readiness pass: LR/LSTM date
# alignment, the future-data-cannot-affect-the-past leakage proof, checkpoint
# train/test-split validation, ensemble degradation when a model fails, and
# export robustness:
pytest tests/test_final_pass.py -v

# Release-hardening regression tests: secret redaction across every
# credential, broker failure handling, live-trading security (disabled by
# default, invalid broker, excessive position, daily limit, duplicate
# orders), numerical edge cases, checkpoint architecture validation, request
# size limits and security headers:
pytest tests/test_hardening.py -v

# Or run everything pytest can collect at once:
pytest tests/ webapp/test_app.py -v
```

**Last full run (V20): 545 passed, 0 failed, 0 skipped, exit code 0**, recorded by `python scripts/release_gate.py --tests-only` in a virtualenv installed from
`requirements-dev-lock.txt` (Python 3.12.3, Linux x86_64, 1 vCPU / 4 GB, CPU only). 188 of those are the original tests and **every pre-existing test file is
byte-identical to V19** (the V17/V18/V19 suites are included unchanged). No test was skipped, weakened or removed. The earlier **user-reported** result of 188 passed
on Windows / Python 3.11.0 applied to V16 and the legacy lock; it has not been re-run on any later version. Full gate results: `docs/V20_ENGINEERING_AUDIT.md` section 8.

## Logging

Server-side diagnostics go through `logging_utils.py` (levels, timestamps,
module attribution, `LOG_LEVEL` in `.env`). Every record passes through a
redaction **filter** attached to the logger itself, not to one handler, so a
secret cannot be written to a log even by a future call site that forgets --
including to a file or aggregator handler a deployment adds later.

The CLI's own `print()` output in `main.py` / `trade.py` / `pipeline.py` is
deliberately left as-is: that is the program's *output*, not diagnostics.

## Numerical robustness

Verified behaviour on degenerate inputs, each covered by a test:

| Input | Behaviour |
|---|---|
| A zero actual price | MAPE excludes it rather than returning `+inf`. Previously one zero made the metric infinite, and that `inf` reached `report.json`, the exports and the API as if it were a measurement. |
| All actual prices zero | MAPE is `NaN` — honestly "not measurable". |
| Constant (flat) price series | All metrics finite. |
| Series shorter than 2 points | Directional accuracy is `NaN`, not a division by zero. |
| Empty dataframe / missing OHLCV columns | Reported as a data-quality **issue**; never silently filled. |
| Too little history | `InsufficientDataError` with an actionable message. |
| Duplicate / unsorted timestamps | Detected and reported; the dataframe is never mutated. |
| Timezone-aware index | Handled. |
| Malformed or empty upstream response | `DataSourceError`. **No fabricated market data, ever.** |

A caveat worth stating: on a perfectly flat series, directional accuracy
reports 100% because `sign(0) == sign(0)`. That is degenerate rather than
meaningful — real price data does not do this, but do not read that number as
skill if you ever construct such a series.

## Security

- **API keys and broker credentials** live only in `.env` (never committed
  -- see `.gitignore`) and `config.py`; they're read server-side and are
  never sent to the browser or embedded in any frontend code.
- **The Gemini key is no longer put in a URL at all.** Google's REST API
  accepts its key either as a `?key=` query parameter or as an
  `x-goog-api-key` request header. The query form meant a failed request's
  exception message contained the full URL -- key included -- verbatim, and
  that message is what gets stored as that provider's "error" field, which
  the webapp displays and which report.json/CSV/PDF exports include. The
  client now sends the key as a **header** by default, removing the exposure
  at its source rather than only scrubbing it downstream. Set
  `GEMINI_AUTH_MODE=query` to restore the old behaviour if you need to.
  **Caveat, stated honestly:** header auth could not be verified against the
  live Google API from the environment this pass was done in (no outbound
  access to it, and no key), so if Gemini starts failing, that escape hatch
  is the first thing to try.
- **LLM provider errors are still redacted before storage or display**, as
  defence in depth. Every provider's error message is redacted
  (`ai_integrations/base_client.py`) against every configured API key before
  it's ever stored, closing this class of leak regardless of which provider
  or URL scheme triggered it -- including any future code path that
  reintroduces a key into a URL.
- **Live trading through the web UI requires a separate operator secret**
  (`WEBAPP_LIVE_TRADING_TOKEN`, minimum 32 characters), fail-closed by
  default. See "Safety model" above for what that protection is and is not.
- **Broker and data-source errors no longer echo upstream response bodies.**
  A second verified leak, of the same class as the Gemini one and found the
  same way: `data_sources/groww_source.py` raised `"Groww token response
  missing 'token': {resp.text}"` — a body from the *authentication* endpoint.
  A token returned under an unexpected key was echoed verbatim into that
  exception (confirmed directly). Upstox's OAuth token exchange did the same,
  and every non-200 path in `trading/brokers/*.py` put `resp.text` straight
  into a message that is written to `outputs/trade_log.csv` and rendered in
  the browser — an upstream body echoing an `Authorization` header reached
  both. Token-endpoint bodies are now never quoted at all; every other
  upstream body goes through `security.safe_upstream_detail()`, which
  redacts and truncates it.
- **Redaction now covers every secret the project holds**, not just the five
  LLM keys (`security.py`): broker credentials, the web-app secret key and
  the operator token, plus pattern-based scrubbing of bearer tokens, `?key=`
  query params and provider key prefixes — so a credential belonging to an
  *upstream* service, echoed back in its own error body, is caught too.
- **Web hardening**: a 256 KB request-body cap (a clean `413`, not a crash),
  `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, a
  `Content-Security-Policy` that permits no external scripts and no framing,
  and `Cache-Control: no-store`. The app **starts with `SystemExit`** if
  `WEBAPP_DEBUG=true` and `ENABLE_LIVE_TRADING=true` are set together —
  Werkzeug's debugger is arbitrary code execution, and a server that can
  place real orders must not run it. **This app has no file-upload routes at
  all**; a test asserts that, so adding one later forces the extension,
  size and traversal handling to be written.
- **Duplicate-order protection** (`DUPLICATE_ORDER_WINDOW_SECONDS`, default
  60s): an identical *live* order — same ticker, side and quantity — is
  refused if one already executed inside the window. It reads the
  append-only audit log rather than in-memory state, so it survives a
  restart and works across separate CLI invocations, which is exactly when
  accidental double-submits happen. Paper orders are never blocked.
- **Checkpoint loading verifies the model itself, not just its metadata.**
  Input shape must be `(None, lookback, 1)` and output shape `(None, 1)`, and
  the scaler must be fitted on exactly one feature. A swapped `model.keras`
  with untouched metadata is refused.
- **No pickle deserialization anywhere in checkpoint loading.** A checkpoint's
  fitted scaler is written and read as plain JSON numbers (`scaler.json`, via
  `_scaler_to_dict()` / `_scaler_from_dict()` in `checkpoints.py`) -- a fitted
  `MinMaxScaler`'s entire state is a handful of numbers (min/scale/data_min/
  data_max/data_range plus a couple of scalar settings), so reconstructing it
  needs no arbitrary-object deserialization at all, and there is no
  code-execution gadget on disk even if a checkpoint directory were ever
  swapped out from under the app. `model.keras` is loaded with
  `tf.keras.models.load_model`, which Keras 3 runs in `safe_mode=True` by
  default (blocking arbitrary Lambda-layer code execution) -- not overridden
  anywhere in this project -- and, as above, both its metadata and its actual
  input/output shapes are validated before reuse. A checkpoint written by an
  older, pickle-based version of this project (`scaler.pkl`) is no longer
  accepted and must be retrained. Checkpoints are still only ever files this
  project wrote, in a directory it owns, under a name that rejects path
  separators and `..` -- **never load a checkpoint obtained from someone
  else.**
- **Unexpected internal errors never echo raw exception text to the
  browser or the JSON API.** Fixed a second, related issue: `/predict`'s
  and `/export/pdf`'s generic exception handlers used to render `str(e)`
  directly into the HTML response; any unexpected internal exception
  (which could in principle contain a file path or other internal detail)
  would have been shown verbatim. Both now show a fixed, generic message
  while the real exception is still printed server-side for debugging. The
  JSON API (`/api/*`) was built with this same principle from the start.
- **Flask config**: session cookies are `HttpOnly` and `SameSite=Lax`
  explicitly (not just relying on framework defaults); `WEBAPP_SECRET_KEY`
  falls back to a random per-process token if unset (never a hardcoded
  value); `debug` mode is off.
- **Live trading requires an explicit, multi-step opt-in** -- see
  "Trading" above; paper trading (no real money, ever) is the default.
- `.gitignore` was added (none existed before) covering `.env`, local
  trading state files (`.paper_portfolio.json`, broker token caches), and
  generated output -- so a real `.env` never ends up in version control by
  accident.

## Reproducibility

**Measured, not assumed.** Running the same prediction twice with identical
input data, configuration and `RANDOM_SEED=1234` produced **bit-identical**
Linear Regression, LSTM and ensemble outputs -- delta exactly `0.0` on all
three. Changing the seed changed the LSTM output, confirming the seed is
genuinely in effect rather than the result being seed-independent. This
specific measurement was taken under an earlier environment (Python 3.12.3, Linux, CPU) and was **not re-measured for V18**:
it predates the V18 dependency re-resolution (numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1). Treat bit-identical reproducibility
as unverified for the current lock until you re-measure it.

This is a measurement on one platform, **not a formal guarantee**. See
`reproducibility.py` for the honest limits: GPU cuDNN kernels, multi-threaded
BLAS reduction order, and any unseeded external source (notably live LLM
responses and market data that the provider may revise after the fact) can
all still introduce variation.


`reproducibility.py` seeds Python's `random`, NumPy, and TensorFlow at the
start of every pipeline run (`RANDOM_SEED` in `.env`, default 42), so
repeated runs on the same data and config produce the same (or
near-identical) results. This is a **best-effort** guarantee, not a formal
one:
- On GPU, some cuDNN/CUDA kernels (certain LSTM/RNN ops, reductions) are
  non-deterministic by default for performance reasons; `TF_DETERMINISTIC_OPS`
  is set to ask TensorFlow to prefer deterministic kernels where available,
  but not every op has one for every TensorFlow/CUDA/cuDNN version.
- On CPU (this project's default, via `tensorflow-cpu`), multi-threaded BLAS
  reduction order can still cause tiny floating-point differences.
- Linear Regression is deterministic by nature regardless of seeding.

See `reproducibility.py`'s module docstring for the full, honest accounting.

Every pipeline run also captures **run metadata** (`reproducibility.capture_run_metadata`),
included in `report.json` (`run_metadata`) and the `/api/predict` response:
ticker, data source, period, random seed, LSTM config (prediction target,
epochs, units, lookback, train/test split), ensemble weighting mode, row
counts (total/train/test), a timestamp, and installed package versions
(numpy/pandas/scikit-learn/tensorflow/yfinance/python). This describes how a
result was produced well enough to reason about reproducing it -- it does
NOT include API keys or credentials, and it can't fully guarantee an
identical re-run given the nondeterminism limits above and the fact that
market data itself can be revised by the provider after the fact.

## The Scilab version

The Scilab implementation (Linear Regression + an AR time-series model, plus
a BUY/SELL/HOLD signal printout) is a **separate, standalone project**:
`stock_prediction_scilab/`. It doesn't depend on anything in this project --
it has its own data fetcher (also covering yfinance/Upstox/Groww) and its
own README. It does not place real orders (see that README for why).

## Design notes / honesty about limitations

- **TensorFlow retracing warnings during walk-forward validation are
  expected, not a bug, and don't affect correctness.** An explicit
  `batch_size=len(X)` on every `.predict()` call (rather than Keras's
  default `batch_size=32`) eliminated the *avoidable* kind -- a test set
  whose length isn't a multiple of 32 getting a differently-shaped final
  partial batch, which used to trigger an extra retrace on every model
  (confirmed: predictions are bit-identical regardless of batching, so this
  was purely a trace-count fix, never a value-changing one). What remains
  is TensorFlow's one-time initial trace for each of the *several distinct
  new Keras models* walk-forward validation legitimately creates (one per
  fold, by design, so each fold is evaluated by a model that only ever
  trained on that fold's own data) -- TensorFlow's own warning message
  names "creating @tf.function repeatedly in a loop" as an expected cause
  for exactly this pattern. Removing this last bit would require reusing a
  single persistent model object across folds instead, which risks subtle
  state leaking between folds if not done with real care -- not a change
  made here, since correctness was prioritized over silencing a cosmetic
  warning (and blindly suppressing it, rather than understanding it, was
  explicitly out of scope for this review).
- **LLMs are not price oracles.** The panel prompt deliberately asks for
  *direction + rough magnitude + reasoning* over a plain-text trend summary,
  not a precise number pulled from nowhere. Treat the panel as a sentiment
  check, not a forecast engine -- it gets the smallest ensemble weight
  (20%, split across however many providers respond) for that reason.
- **Past performance ≠ future results.** Both LR and LSTM are fit purely on
  historical price patterns; neither knows about news, earnings, or macro
  events unless an LLM happens to mention them from its own training data
  (and even then, not reliably or currently).
- **Chronological splits only.** Both models use a strict time-ordered
  train/test split -- shuffling would leak future data into training and
  produce misleadingly good metrics.
- **The BUY/SELL/HOLD rule is intentionally simple and transparent.** It's a
  threshold on the ensemble's predicted % change, not a black box -- you can
  read the entire rule in `trading/signal_generator.py` in about 10 lines.
- **The daily trade limit only counts actually-executed trades**, not
  blocked or rejected attempts -- a misconfigured order shouldn't lock you
  out of trading for the rest of the day.
- **BUY and SELL orders are sized differently, on purpose.** A BUY is sized
  against available capital (how much can I afford). A SELL is sized against
  your actual held position (how much do I own) and defaults to closing the
  full position on a SELL signal -- it can never be sized against cash,
  since selling doesn't cost you capital, and a SELL signal with zero shares
  held is correctly blocked rather than attempting a phantom short-sale.
- **Walk-forward validation and the backtest are both additions in this
  pass, not audits of prior features** -- there was no backtesting engine in
  this project before `backtest.py`, and no expanding-window validation
  before `validation.py`. Both are opt-in (`--walk-forward` / `--backtest`)
  because they cost meaningful extra runtime (the LSTM gets retrained
  several more times).
- **Stop-loss/take-profit and the daily loss limit only work with the paper
  broker today.** A live broker would need its own cost-basis/position
  lookup to support the same checks -- see "Risk controls" above.
- **Dynamic ensemble weighting never increases the LLM panel's influence.**
  It only reallocates the LR/LSTM split based on walk-forward performance;
  the LLM panel's weight is fixed at whatever `.env` configures, by design.
- **The backtest's "trade fills at that day's own close" assumption is a
  simplification**, common in lightweight daily backtests but not a claim of
  intraday realism -- see `backtest.py`'s module docstring for the full
  reasoning.
- **The "next day" prediction bug (see "LSTM: price vs. return target"
  above) was likely the most consequential finding of this review.** It
  affected both LR and LSTM, predates the LSTM/ensemble work entirely, and
  was silent -- nothing crashed, the reported prediction just quietly never
  reflected the actual most recent trading day's data. It was found by a
  deliberate, direct test (shock the last row's Close by a large amount and
  check whether the reported prediction changes at all), not by inspection
  alone -- worth keeping that kind of test around for any future change to
  either model's data-preparation path.
- **Model checkpointing (`checkpoints.py`) only covers the LSTM.** Linear
  Regression trains fast enough (well under a second on typical data sizes)
  that checkpointing it wouldn't meaningfully help; the LSTM is the
  expensive one. A checkpoint is refused (not silently reused) on any
  ticker/lookback/prediction-target mismatch -- see "Model checkpointing"
  above -- but a `units` mismatch alone is currently still refused too
  (rather than, say, allowed with a warning), which is the conservative,
  safe default at the cost of occasionally retraining when it technically
  could have reused a compatible-enough checkpoint.
- **Gemini's API key is still sent as a URL query parameter** (Google's
  documented REST auth scheme for this endpoint), not switched to
  header-based auth, because verifying that alternative actually works
  would require live network access to Google's API that wasn't available
  while reviewing this. The actual leak this created (the key appearing in
  error messages shown in the webapp and written into exported reports) is
  fixed via a verified, provider-agnostic redaction layer in
  `ai_integrations/base_client.py` regardless -- but moving this specific
  call to header-based auth (if confirmed still supported) would remove the
  leak at its source rather than only redacting it after the fact, and
  would be a good follow-up.
- **This was developed and tested in a sandboxed environment with
  restricted outbound network access.** All automated tests use synthetic
  data or mocked network calls and don't depend on live internet access.
  Live, network-connected CLI runs were attempted at multiple points during
  development; the exact failure mode has varied by session -- at one point
  an installed `yfinance`/`curl_cffi` combination hit a
  `ImpersonateError('Impersonating chrome150 is not supported')` (a
  browser-TLS-fingerprint impersonation compatibility issue), and in the
  most recent pass Yahoo Finance instead returned "no price data found" for
  well-known, highly liquid tickers (AAPL, MSFT) that unquestionably have
  data -- consistent with this sandbox's outbound network path being
  rate-limited or blocked by Yahoo's anti-bot measures, not a code defect
  (the retry/fallback logic itself ran correctly both times, trying every
  configured source/suffix and producing a clear, accurate final error
  rather than crashing or silently returning wrong data). If you see either
  of these locally, try upgrading `curl_cffi` and `yfinance` together,
  wait and retry (transient rate-limiting), or use `--source upstox` /
  `--source groww` instead. Every feature added or changed in this and
  prior review passes (dynamic weighting, the LSTM return-mode, the
  next-day-prediction fix, confidence/uncertainty, the backtest metrics and
  weighting-consistency fix, the JSON API, and checkpointing) was instead
  verified against synthetic data run through the real, unmodified
  pipeline/webapp code paths -- the same approach this project's own test
  suite already used before this review, not a weaker/different one
  introduced for it.
