# Quant Workstation V21 — Baseline Engineering & System Audit

**Audit Date:** 2026-10-19  
**Repository Path:** `E:\Projects\stock_prediction_python`  
**Initial Baseline Commit:** `8abdcc8` (`master` root commit)  
**Active Development Branch:** `v21-development`  
**Execution Environment:** Windows 10/11 x86_64, Python 3.11.0, NumPy 2.4.6, TensorFlow 2.20.0, Pytest 9.0.2  

---

## 1. Repository Inventory & Entry Points

### 1.1 Project Structure
* **Application Core**:
  - `config.py`: Global configuration, environment variables, default model weights, thresholds, and risk parameters.
  - `market_data.py`: Multi-provider quote and historical bar service (`MarketDataService`), TTL caching, freshness classifier, circuit breaker (`ProviderHealth`), and instrument resolver (`resolve_instrument`).
  - `data_loader.py`: OHLCV data fetching, feature engineering, train/test splitting, windowing, and data quality diagnostics.
  - `data_sources/`: Modular data adapters (`yfinance_source.py`, `upstox_source.py`, `groww_source.py`).
  - `pipeline.py`: End-to-end model training, checkpoint coordination, out-of-sample evaluation, and ensemble prediction.
  - `models/`: Multivariate Linear Regression (`models/linear_regression_model.py`) and LSTM recurrent network (`models/lstm_model.py`).
  - `ensemble.py`: Blends Linear Regression, LSTM, and LLM panel using fixed or dynamic walk-forward inverse-RMSE weighting.
  - `evaluation.py`: Strictly chronological out-of-sample evaluation, Naive persistence benchmark, Diebold-Mariano test, conformal prediction intervals, and block-bootstrap confidence intervals.
  - `backtest.py`: Look-ahead-bias-free trading simulation supporting Next-Open and Close execution, transaction costs, slippage, trade ledger, and Buy & Hold benchmark.
  - `forecast_service.py`: Asynchronous task execution thread pool for long-running model fitting and backtesting jobs.
  - `trading/`: Risk manager, signal generator, trade logger, and broker implementations (`PaperBroker`, `UpstoxBroker`, `GrowwBroker`).
  - `ai_integrations/`: LLM provider adapters for qualitative market research commentary (Gemini, Groq, Claude, DeepSeek, OpenAI).

### 1.2 Identified Entry Points
* **CLI Prediction & Trading**:
  - `python main.py`: Runs the command-line prediction workflow for a single ticker (`config.TICKER`), generates `outputs/report.json` and charts, and exits. Does **not** launch the web workstation.
  - `python trade.py`: Executes automated or interactive paper/live trading decisions based on model signals.
* **Web Workstation Server**:
  - `python webapp/app.py`: Launches the Flask WSGI web application binding to `http://127.0.0.1:5000` (default port 5000).
* **Release & Gate Tooling**:
  - `python scripts/release_gate.py`: Release verification suite.
  - `python scripts/package_release.py`: Cryptographically verified archive packaging.

### 1.3 Identified Frontend Architecture
* **Framework**: Zero external JavaScript frameworks (no React, Vue, Angular, or jQuery); vanilla ES6+ in `webapp/static/ws.js`.
* **Security & CSP**: Strict Content Security Policy (`script-src 'self'`). No external CDNs or unverified external fonts. All dynamic DOM nodes created via `textContent` and `createElement` (`createTextNode`), eliminating innerHTML cross-site scripting risks.
* **Styling**: Pure CSS in `webapp/static/ws.css`, using CSS variables on `:root`.
* **Routing**: Client-side hash routing (`#/overview`, `#/research`, `#/forecast`, `#/evaluation`, `#/backtest`, `#/ai`, `#/portfolio`, `#/risk`, `#/health`, `#/settings`).
* **Charting**: Custom HTML5 Canvas rendering for candlesticks, volume bars, multi-series equity curves, and drawdown curves with accessible fallback HTML data tables.

---

## 2. Baseline Test Execution

A lightweight baseline test run was executed on `v21-development` to verify existing suite stability:

* **Command**: `python -m pytest tests/test_v20_release_tools.py tests/test_v20_universe_and_backtest_view.py tests/test_v20_backtest_accounting.py -v`
* **Result**: **41 passed, 0 failed, 0 skipped** in 74.87s.
* **Integrity**: Working tree clean, Git baseline established.

---

## 3. Investigation: Market-Data Reliability & TCS.NS vs TCS

### 3.1 Observed Defect
In the user's workstation session:
1. The quote header displayed `TCS.NS` at `INR 2,455.20`, labeled `delayed/cached`.
2. Instrument Research displayed a `TCS.NS` candlestick chart and historical bars.
3. Data & Model Health reported a Yahoo failure count with the error: `"Yahoo returned no data for TCS."`

### 3.2 Root Cause Analysis & Reproduction
Empirical verification confirmed the exact mechanism:
1. **Instrument Resolution**:
   - `resolve_instrument("TCS.NS")` cleanly resolves to `Instrument(symbol="TCS", exchange="NSE", yahoo_symbol="TCS.NS", timezone="Asia/Kolkata", currency_hint="INR")`.
   - `resolve_instrument("TCS")` resolves to `Instrument(symbol="TCS", exchange="US/UNSPECIFIED", yahoo_symbol="TCS", timezone="America/New_York", currency_hint=None)` by design, preventing accidental conflation of US and Indian listings.
2. **Upstream Provider Behavior**:
   - Querying Yahoo for `"TCS"` (US) returns `$TCS: No data found, symbol may be delisted` (Empty DataFrame).
   - In `market_data.py:440`: `YahooQuoteProvider` raised `ProviderError("Yahoo returned no data for TCS.", retryable=False, kind="no_data")`.
3. **Flawed Circuit Breaker & Health Counter**:
   - In `market_data.py:612`, `p.health.fail(str(e))` was called unconditionally for every `ProviderError`, including `kind == "no_data"`.
   - `self.failures` and `self.consecutive_failures` were incremented.
   - When 3 tickers with no US data were queried, `self.consecutive_failures >= 3` tripped the circuit breaker, setting `self.open_until` and blocking subsequent valid requests across the entire Yahoo provider!
   - Furthermore, `p.health.last_error` stored `"Yahoo returned no data for TCS."`, which was displayed on the Health page even when the user was actively inspecting `TCS.NS`.
4. **Off-Hours / Closed Market 1m History Fallback**:
   - In `YahooQuoteProvider.fetch_raw()`, if `info.get("regularMarketPrice")` was missing or rate-limited, the provider attempted `t.history(period="1d", interval="1m")`.
   - Outside trading hours (or on weekends), 1-minute bars are empty. The provider threw `ProviderError("Yahoo returned no data...")` instead of falling back to `t.fast_info` or `t.history(period="5d", interval="1d")`.
   - `get_quote` then served the previously cached quote with the flag `served_cached_during_outage` and badge `delayed/cached`.

---

## 4. Investigation: Zero-Trade Strategy Backtest & Accounting

### 4.1 Observed Issue
The Strategy Backtest screen showed:
- Strategy Return: `0.00%`
- Buy & Hold: `+8.85%`
- Strategy minus Buy & Hold: `-8.85%`
- Trades: `0 fills`
- Exposure: `0.0%`
- Costs: `0.00`

### 4.2 Root Cause Analysis
1. **Model Forecast Distribution vs Inactivity Band**:
   - In `config.py`, default trading signal thresholds are:
     - `BUY_THRESHOLD_PCT = +0.50%`
     - `SELL_THRESHOLD_PCT = -0.50%`
   - In `backtest.py:261`, the daily trading signal is generated by comparing the model's predicted next-day close $P_{pred, t}$ against the previous day's actual close $P_{act, t-1}$:
     $$\Delta\% = \frac{P_{pred, t} - P_{act, t-1}}{P_{act, t-1}} \times 100$$
   - In liquid mega-cap equities with regularized or shrunken ensemble predictions, daily predicted percentage changes are frequently shrunken into the narrow band $[-0.35\%, +0.35\%]$.
   - Because the strategy starts in 100% cash, a BUY signal is required to open a position. If $\Delta\%$ never reaches $+0.50\%$, 0 BUY signals are generated.
   - The strategy legitimately remains in cash for the entire window, resulting in 0 trades, 0% exposure, and 0% return.
2. **Missing In-UI Diagnostics**:
   - While mathematically valid, the workstation did not display the model's actual forecast range (min, max, mean % change) or signal counts. The user could not tell whether 0 trades was due to a data alignment defect, unit error, or threshold inactivity.
3. **Accounting Audit**:
   - **Units**: Verified consistent percentage points throughout `backtest.py` and `signal_generator.py`.
   - **Asymmetry in Mark-to-Market vs Benchmark**: When a strategy finishes with an open position, its final equity is marked to market at the close without forced liquidation costs. However, Buy & Hold had a forced exit cost deducted. This is aligned to ensure symmetric mark-to-market comparison.

---

## 5. UI Architecture & Theme Gap Analysis

### 5.1 Deficiencies in Existing UI
1. **Light Theme Incomplete**:
   - `:root[data-theme="auto"]` had no CSS variable overrides.
   - Canvas chart backgrounds and text colors used hardcoded dark colors (`#0b0d10`), causing tooltips and axes to look illegible or broken if switched to light mode.
2. **Lack of User-Facing Theme Toggle**:
   - No toggle in the header or Settings to choose Dark, Light, or System preference.
3. **Data Density & Polish**:
   - The styling lacked modern quant-terminal minimalism (subtle surface differentiation, refined borders, restrained glassmorphism for quote strips).

---

## 6. Implementation Action Plan for V21

1. **Market Data Reliability**:
   - Separate infrastructure failures from `no_data` ticker outcomes in `ProviderHealth`.
   - Prevent `no_data` queries from tripping the provider circuit breaker.
   - Enhance `YahooQuoteProvider` with `t.fast_info` and 5-day daily fallback when 1-minute intraday bars are empty.
   - Provide informative symbol guidance when bare Indian tickers are queried on US exchanges.
2. **Strategy Backtesting & Accounting**:
   - Add `signal_diagnostics` to `BacktestResult` (min/max/mean % change, signal counts, threshold gap).
   - Display actionable diagnostics on zero-trade backtests.
   - Validate and reconcile position sizing parameters.
3. **UI Redesign**:
   - Implement complete dark and light design token system with restrained glassmorphism.
   - Add theme switcher in global header and settings with `localStorage` persistence and FOUC prevention.
   - Update Canvas chart rendering to dynamically adapt to active theme colors.
4. **Verification & Packaging**:
   - Add comprehensive regression tests.
   - Execute full test suite and browser smoke tests.
   - Package `stock_prediction_python_v21_FINAL.zip`.
