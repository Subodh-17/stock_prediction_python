# Quant Workstation V21 — Runtime QA & Verification Report

## Overview
This document records the exact runtime environment, verification commands, exit codes, and test outputs for the **Quant Workstation V21** release candidate.

Verification executed on: Windows 11 (AMD64), Python 3.11.0.

---

## 1. Environment & Dependency Audit

### Python Runtime
- **Command**: `python --version`
- **Exit Code**: `0`
- **Output**: `Python 3.11.0`

### Dependency Integrity (`pip check`)
- **Command**: `python -m pip check`
- **Exit Code**: `0`
- **Execution Time**: 10.6s
- **Output**: `No broken requirements found.`

### Deep Learning Runtime Check
- **Command**: `python -c "import tensorflow as tf, keras; print('tensorflow', tf.__version__, 'keras', keras.__version__)"`
- **Exit Code**: `0`
- **Execution Time**: 7.8s
- **Output**: `tensorflow 2.21.0 keras 3.15.1`

### Bytecode Compilation
- **Command**: `python -m compileall -q .`
- **Exit Code**: `0`
- **Execution Time**: 0.4s
- **Output**: Clean compilation across all repository files.

---

## 2. Server Startup & Endpoint Smoke Test

- **Command**: `python scripts/release_gate.py --skip-tests`
- **Server Execution**: Bound to `127.0.0.1` on dynamic port.
- **Warmup**: `WARMUP_ON_START=false` for clean cold-start verification.
- **Exit Code**: `0`
- **Startup Latency**: 9.5s
- **Endpoint Responses**:
  - `GET /` $\rightarrow$ `200 OK`
  - `GET /api/v2/meta` $\rightarrow$ `200 OK`
  - `GET /api/v2/health` $\rightarrow$ `200 OK`
  - `GET /api/v2/ai/status` $\rightarrow$ `200 OK`
  - `GET /static/ws.js` $\rightarrow$ `200 OK`
  - `GET /static/ws.css` $\rightarrow$ `200 OK`

---

## 3. Targeted Regression Test Results

### 3.1 Market Data Reliability (`tests/test_v21_market_data_reliability.py`)
- **Command**: `pytest tests/test_v21_market_data_reliability.py`
- **Results**: 4 passed in 1.41s
- **Checks Verified**:
  - Unknown symbol lookup records `no_data` without tripping provider circuit breaker.
  - Bare `TCS` returns non-US exchange hint while preserving provider availability.
  - Subsequent query to `TCS.NS` succeeds with provider circuit remaining closed.
  - Contract preservation: `resolve_instrument("TCS")` remains `US/UNSPECIFIED` by default.

### 3.2 Backtest Zero-Trade Diagnostics (`tests/test_v21_backtest_diagnostics.py`)
- **Command**: `pytest tests/test_v21_backtest_diagnostics.py`
- **Results**: 2 passed in 1.49s
- **Checks Verified**:
  - When daily predictions fall into $[-0.35\%, +0.35\%]$, backtest produces complete `signal_diagnostics` dict with `zero_trade_reason`, `min_pct`, `max_pct`, and `buy_signals=0`.
  - Position sizing parameter $> 1.0$ (e.g. `100` from web UI) is normalized to fractional `1.0` (100% of equity).

### 3.3 Static UI & WCAG AA Contrast (`tests/test_v18_ui.py`)
- **Command**: `pytest tests/test_v18_ui.py`
- **Results**: 7 passed in 0.31s
- **Checks Verified**:
  - All text colors (`tx`, `mut`, `dim`, `acc`, `gain`, `loss`, `info`, `warn`) on all surfaces (`bg`, `panel`, `raise`) meet WCAG AA ($\ge 4.5:1$).
  - Form control borders meet $\ge 3.0:1$ non-text contrast against surfaces.
  - Focus indicator (`:focus-visible`) uses high-contrast accent token.
  - Accessible table alternatives present for line, histogram, and candlestick charts.
  - Direction never conveyed by color alone (always paired with glyph `▲`, `▼`, `■` or label).

### 3.4 Forecast Marker Isolation (`tests/test_v19_ui.py`)
- **Command**: `pytest tests/test_v19_ui.py`
- **Results**: 6 passed in 0.28s
- **Checks Verified**:
  - Forecast drawn in dedicated fixed-pixel right-hand zone as dashed whisker/diamond.
  - Historical bars series is never mutated or extended with forecasts.
  - Accessible name clearly identifies model estimate vs observations.
  - Missing prediction intervals honestly flagged as unavailable.

### 3.5 Web Application Security & API Suite (`tests/test_v17_webapp.py`)
- **Command**: `pytest tests/test_v17_webapp.py`
- **Results**: 30 passed in 42.59s
- **Checks Verified**:
  - All 10 views rendered with CSP `script-src 'self'` and `X-Content-Type-Options: nosniff`.
  - CSRF protection enforced on all mutating endpoints.
  - No route allows placing live broker orders (`live_orders_via_this_api: false`).
  - Strict input sanitization (path traversal, SQL injection patterns return 400).
  - Rate limiting active with `Retry-After` headers.
  - Paper trading verified with simulation badges and realistic cash deduction.
  - Zero `innerHTML`, `document.write`, or `eval()` in frontend script.

### 3.6 Universe Evaluation & View Honesty (`tests/test_v20_universe_and_backtest_view.py`)
- **Command**: `pytest tests/test_v20_universe_and_backtest_view.py`
- **Results**: 2 passed in 61.27s
- **Checks Verified**:
  - Real expanding-window universe evaluation across multiple instruments.
  - Backtest view explains cash-only or low-exposure results without false claims of skill.

### 3.7 Full Release-Gate Test Suite
- **Command**: `python scripts/release_gate.py --tests-only`
- **Test Command**: `python -m pytest tests webapp/test_app.py -q -p no:cacheprovider --ignore=tests/qa_harness.py`
- **Exit Code**: `0` (PASS)
- **Pytest Reported Time**: 1161.45s (19m 21s)
- **Total Wall-Clock Time**: 1193.63s
- **Test Counts**:
  - **Passed**: 551
  - **Failed**: 0
  - **Errors**: 0
  - **Skipped**: 0
- **Summary**: 100% pass rate across all 551 tests spanning unit tests, regression suites, pipeline modeling, walk-forward validation, API endpoints, paper trading, and UI accessibility.

---

## 4. Security & Safety Verification

1. **Broker Isolation**: Verified that live trading remains completely disabled by default. No live execution endpoints exist in the API (`api_v2.py`).
2. **Credential Redaction**: Verified that no real API keys (`AIza...`, `sk-...`, `gsk_...`) or secret tokens exist in code, documentation, or static assets.
3. **Template Integrity**: `.env.example` verified to contain placeholders only.
4. **Input Defense**: All symbol parameters pass through strict regex whitelist `^[A-Za-z0-9_.:\-^]{1,24}$`.
