# V17 change record

This document says what changed, what was actually verified, and what was **not**. Where a claim could not be
verified it is marked **UNVERIFIED**.

## 1. What could and could not be verified

The development sandbox could reach PyPI but **not** Yahoo Finance, Google (Gemini), Groq, Upstox or any other market-data
or AI endpoint (checked directly; connections refused). Consequently:

| Item | Status |
|---|---|
| Any live market-data call (Yahoo / Upstox / Groww) | **UNVERIFIED.** Never executed. Covered by mocked tests only. |
| Any live AI call (Gemini / Groq / OpenRouter / OpenAI / Anthropic / xAI / DeepSeek) | **UNVERIFIED.** Never executed. |
| Gemini 404 root cause | **Verified from Google's official deprecations page** (fetched 2026-10-11, page dated 2026-10-09): `gemini-2.0-flash` had a shutdown date of 2026-06-01. The old default pointed at a retired model. |
| Gemini free-tier models | From Google's official pricing page (same date): a free tier is listed for several Flash/Flash-Lite models; the 2.5 family is restricted to accounts that already used it. Quotas are per project and **not** quoted here -- read them in AI Studio. |
| Groq / OpenRouter free tiers | **UNVERIFIED** (not researched against current docs). The adapters are OpenAI-compatible and ship with **no default model**; "Test connection" lists the models your key can use. |
| Upstox quote adapter | **UNVERIFIED** against the live API. Written from the documented response shape; real-time entitlement depends on your account. |
| Groww | History fallback only (unchanged). No quote adapter was added. |
| Full browser walk-through | **Done** with Chromium (Playwright), desktop 1440x900 and mobile 390x844, **using the QA harness** (`tests/qa_harness.py`): real app, real pipeline, real LR+LSTM training, real evaluation/backtest code -- but with **synthetic fixture prices** and a **fake HTTP layer** for Gemini. Screenshots from it are test artifacts, not market data. In particular the "verified" Gemini badge seen there proves the UI flow, not a real Gemini call. |

Consequence for "Live": with the default configuration **no provider can ever produce a `live` quote** (Yahoo via yfinance is
never classified live; Upstox only if you have a valid token and the response carries a fresh trade timestamp). That is
intentional.

## 2. Audit findings

### Bugs fixed
| # | Finding | Fix |
|---|---|---|
| 1 | Gemini default model retired (404) and error bodies discarded, so a 404 looked generic | New default; `classify_http` reads status **and** body; `Test connection` lists models the key can use |
| 2 | `cp .env.example .env` made the web app 500 on every page (blank `WEBAPP_SECRET_KEY=` -> empty Flask secret) | `getenv(...) or token_hex(32)`; regression test runs the app in a subprocess with the blank value |
| 3 | LLM panel "ceiling" did not bound latency (1 s ceiling, 6 s actual): `with ThreadPoolExecutor` waits for all threads | `shutdown(wait=False)` + `wait(FIRST_COMPLETED)` with a real deadline; stragglers abandoned |
| 4 | Data layer had no quote, timestamp, freshness or provenance; orders were priced from the last *historical close* | `market_data.py` (Quote + freshness policy); paper orders require a usable quote |
| 5 | LLM panel held a fixed 20 % weight in the numeric price | Default 0; commentary only (see 4.3) |
| 6 | No naive/drift baseline anywhere; R² on price levels flatters any model | `evaluation.py`, same-window comparison |
| 7 | "Confidence" = +/-RMSE band; model agreement read as certainty | Split-conformal interval with measured held-out coverage, or "Insufficient evaluation data"; legacy heuristic kept but labelled |
| 8 | `.BO` tickers resolved on NSE in the Upstox/Groww adapters; bare tickers silently got `.NS`/`.BO` suffix fallbacks | New path uses an explicit `Instrument(exchange, symbol)`; bare symbol stays `US/UNSPECIFIED` |
| 9 | Forecast ran inside the Flask request (LSTM training blocks a worker); `/api/*` unthrottled and unauthenticated | Background job service; per-IP rate limit on `/api/v2` **and** legacy `/api/*`, `/predict` |
| 10 | The still-open day's partial bar was treated as a settled close | Dropped before modelling when the session is open |
| 11 | Backtest marked positions to the fill price and had no slippage/next-open mode/drawdown or aligned benchmark curve | Added `execution="next_open"`, `slippage_bps`, mark-to-close, `drawdown_curve`, `benchmark_curve` with the same entry rule and costs |
| 12 | README presented results from synthetic data (12-run regime comparison, `sample_outputs/`) | Annotated / removed |

### Found, not fixed (be aware)
* The classic `/predict` flow and the CLI still use the legacy fetch (with its suffix fallback) and run synchronously.
* **Checkpoint contamination is not enforced by metadata.** Reasoning shows a reused checkpoint cannot overlap the new test
  window when data only moves forward, and the auto checkpoint name includes instrument/period/target/lookback -- but the
  training end date is not stored and compared. A test that checks this would be the next step.
* Walk-forward LSTM folds use fewer epochs than the main model.
* The LLM's incremental value **could not be evaluated**: no historical, point-in-time LLM forecasts exist, and generating
  them now would use hindsight. It is therefore excluded rather than "tested".
* `requirements-lock.txt` was not regenerated. This build was tested on Python 3.12.3 with numpy 2.4.4, pandas 3.0.2,
  scikit-learn 1.8.0, tensorflow 2.21.0, yfinance 1.7.0, flask 3.1.3, pytest 9.1.1 (lock pins numpy 2.4.6 / pandas 3.0.5).

## 3. Architecture additions

```
market_data.py        Instrument, Quote, classify_freshness, providers (Yahoo, Upstox), TTL cache, single-flight,
                      backoff+jitter+Retry-After, circuit breaker, health + measured latency
news_feed.py          RSS/Atom from NEWS_FEEDS only; DOCTYPE refused; dedupe; honest empty states
indicators.py         SMA, EMA, Bollinger, RSI (Wilder), MACD (RSI checked against the textbook example: 70.46)
evaluation.py         Aligned frame, naive/drift/LR/LSTM/ensemble, metrics with explicit "unavailable", DM, residuals, conformal
forecast_service.py   Background jobs, dedupe, result cache keyed by last bar, partial-bar removal, backtest on the same frame
research.py           Fact pack -> JSON schema -> server-side validation (unknown citations dropped, unverified numbers flagged)
ai_integrations/providers.py   Unified adapters, error classes, cooldowns, concurrent first-valid-wins with a global budget
webapp/api_v2.py      JSON API (CSRF header, per-IP rate limit, no live-order route)
webapp/templates/workstation.html, static/ws.css, static/ws.js   The UI (no framework, no CDN, no innerHTML)
```

### Freshness policy (the one place it is decided: `classify_freshness`)
`live` needs (provider entitled to real-time) AND (source timestamp <= `LIVE_MAX_AGE_SECONDS`). A provider claiming
"realtime" without entitlement is *not* live. Known delay -> `delayed` (becomes `stale` past delay + grace). Daily data ->
`end_of_day`. Unknown delay or no source timestamp -> `unverified`. Future timestamps are distrusted. A recent
**retrieval** time never changes the category. Cached quotes keep ageing from their *source* time. Fallbacks to a
lower-freshness provider are recorded in `selection_trail` / `downgraded_from` and shown in the UI.

## 4. Methodology

### 4.1 Evaluation (`evaluation.py`)
* One set of out-of-sample target dates shared by every model; target = next-trading-day close (price) and close-to-close
  return; adjusted closes. A guard raises if model outputs and raw closes are misaligned.
* Baselines: **naive** (yesterday's close) and **drift** (mean daily return of data *strictly before* the first test date;
  undefined with < 30 prior returns).
* Metrics: MAE, RMSE, return-RMSE, MAPE (undefined near zero), directional accuracy vs the previous close (flat days
  excluded), RMSE improvement vs naive, approximate Diebold-Mariano z (no autocorrelation correction -- indicative only),
  residual mean/std/skew/lag-1 autocorrelation. Anything undefined is `null` + a reason, never 0. `n_obs` and the period
  travel with every table.
* The LR/LSTM `lr_lstm_ensemble` row uses the configured LR:LSTM ratio, fixed (no dynamic weights).

### 4.2 Uncertainty
Split-conformal interval on absolute return residuals of the ensemble: the quantile comes from the first half of the
out-of-sample residuals, **its coverage is measured on the second half and displayed**, the published interval uses all
residuals. Requires >= 80 out-of-sample observations, else "Insufficient evaluation data". It assumes roughly exchangeable
residuals, which fails in volatility clusters (a test demonstrates coverage collapsing under a regime change, and the UI
shows the measured gap).

### 4.3 Ensemble and LLMs
Numerical forecast = LR + LSTM (50/50 default, env-configurable). LLMs contribute **commentary only**. If you set
`LLM_PANEL_WEIGHT > 0` the legacy path still honours it, but nothing in V17 evaluates or endorses that.

### 4.4 Backtest assumptions
Signal for day *i* uses data through day *i-1*. Default fill: **day *i* open** (first price tradable after the signal);
`close` mode (market-on-close, optimistic) is available and is the library default for backward compatibility. Costs per
side + slippage bps on every fill; a missing/invalid open means **no fill** (counted as `skipped_fills`). Long-only,
whole-position entries, marked to market at each close, adjusted prices, no taxes, no partial fills, single instrument.
The window is the models' held-out period; thresholds tuned on it are in-sample.

## 5. Measured numbers (CPU sandbox, **fixture inputs**, not real-market latency)
| Measurement | Result |
|---|---|
| Quote cache: simulated 300 ms upstream | miss 300 ms; hit median 0.018 ms (p95 0.029 ms); 201 requests -> 1 upstream call |
| Panel ceiling 1.0 s with a 6 s provider | before 6.0 s; after 1.0 s |
| Forecast job (520 bars, 3 epochs) | first run 6.4 s total (models 3.5 s), submit call returned in 0.78 s (first-call imports); same bar again: 0 ms (cache hit, no retraining) |
| Same bar, result cache cleared, checkpoint kept | 0.7 s; LSTM "reused checkpoint" |
| In-browser run with walk-forward enabled | ~27 s, UI stays responsive (training is off the request thread) |

Real provider latency was **not** measurable here.

## 6. Test results
`python -m pytest tests webapp/test_app.py -q` -- see the final report for the exact counts of the last run. Baseline before
any change: **188 passed (159 s)**. No existing test was edited, skipped or deleted. Mutation checks (breaking the
next-open fill; removing the real-time-entitlement check) made the new tests fail as intended.
Tests that need real credentials/network were not run.

## 7. Known limitations
* No provider was exercised live (section 1). Treat every provider integration as "implemented and mock-tested".
* Fonts are the system stack (the CSP forbids external fonts); numerals use tabular figures.
* Charts have `aria-label` summaries and a mouse tooltip, but **no keyboard-accessible data table** yet; colour contrast was
  designed (text paired with glyphs; hollow vs filled candles) but not machine-audited.
* The new UI does not display real broker balances; live orders remain only in the gated classic flow.
* Market breadth is shown as "not available" (no breadth source); news requires you to configure feeds whose terms allow it.
* Watchlist/preferences live in browser `localStorage` (non-secret by design).
* Daily bars only: intraday history is not supported in this build.

## 8. Windows PowerShell setup
```powershell
cd stock_prediction_python
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r webapp/requirements.txt      # runtime
pip install -r requirements-dev.txt                              # + pytest, only if you will run tests
Copy-Item .env.example .env                                      # then edit; no key is required to start
python -c "import secrets; print(secrets.token_hex(32))"         # optional: paste into WEBAPP_SECRET_KEY
python webapp\app.py                                             # http://127.0.0.1:5000 (local only by default)
```
Free AI (optional): create a key in Google AI Studio, set `GOOGLE_API_KEY=...` in `.env`, restart, open **AI Research** and press
**Test connection** (it lists the models your key can use; set `GOOGLE_MODEL` to one). Optional news: set `NEWS_FEEDS`.
Indian instruments: `NSE:RELIANCE`, `BSE:RELIANCE`. International: `VOD.L`, bare `AAPL` for US.

## 9. Environment variables (new in V17; all optional)
`GROQ_API_KEY GROQ_MODEL OPENROUTER_API_KEY OPENROUTER_MODEL AI_REQUEST_TIMEOUT_SECONDS AI_GLOBAL_BUDGET_SECONDS
AI_FAILURE_COOLDOWN_SECONDS AI_RESEARCH_CACHE_SECONDS LIVE_MAX_AGE_SECONDS QUOTE_TTL_SECONDS HISTORY_TTL_SECONDS
DELAYED_STALE_AFTER_SECONDS PROVIDER_MAX_CONCURRENCY FORECAST_CACHE_TTL_SECONDS NEWS_FEEDS NEWS_TIMEOUT_SECONDS
WEBAPP_EXPENSIVE_RPM` -- documented in `.env.example`. Changed defaults: `GOOGLE_MODEL`, `LR_WEIGHT`=0.5, `LSTM_WEIGHT`=0.5,
`LLM_PANEL_WEIGHT`=0.
