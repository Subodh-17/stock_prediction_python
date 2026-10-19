# Quant Workstation V21 — Changelog

## Release Summary
**Version**: Quant Workstation V21 — Production-Quality Engineering, Bug Fixes & UI Redesign  
**Release Date**: October 2026  
**Status**: Release Candidate (Verified)

Quant Workstation V21 addresses critical operational defects in market data fetching, quantitative backtesting diagnostics, cross-platform file encoding, and visual design. V21 delivers a redesigned quantitative terminal aesthetic with dark and light themes, anti-FOUC execution, verified WCAG AA contrast, and transparent signal accounting for zero-trade backtests.

---

## 1. Market Data Reliability & Provider Health

### Defect: Inappropriate Circuit Breaker Tripping on Missing Tickers
- **Root Cause**: When a user queried an invalid or unlisted ticker (e.g. bare symbol `TCS` which resolves to US exchanges by default), Yahoo returned a ticker-level "no data" response. The provider health monitor previously called `p.health.fail()` unconditionally, treating client-level 404s as upstream infrastructure outages. After 3 requests, the circuit breaker tripped open, blocking subsequent valid requests (e.g. `NSE:TCS` or `TCS.NS`).
- **Resolution**:
  - Added `record_no_data()` to `ProviderHealth` to track ticker misses separately without incrementing infrastructure failure counters or opening the circuit breaker.
  - Updated `YahooQuoteProvider` to invoke `record_no_data()` on symbol lookup misses and only invoke `p.health.fail()` on true network/HTTP connection errors.
  - Preserved the existing contract where bare symbols default to `US/UNSPECIFIED` (as verified by `test_v17_market_data.py`).

### Defect: Intraday 1-Minute Bar Emptiness Outside Market Hours
- **Root Cause**: When fetching quotes outside active trading hours, 1-minute intraday bars are often empty from Yahoo Finance, triggering spurious `no_data` errors.
- **Resolution**:
  - Implemented robust multi-tiered fallback in `YahooQuoteProvider.fetch_raw`:
    1. Primary: 1-day 1-minute intraday bars.
    2. Fallback 1: `ticker.fast_info` (`last_price`, `previous_close`, `currency`, `timezone`).
    3. Fallback 2: 5-day daily bars to retrieve the latest completed trading session.
  - Enhanced error messages for bare symbols with non-US hints (e.g. `NSE:TCS` or `TCS.NS`).

---

## 2. Strategy Backtesting & Zero-Trade Signal Accounting

### Defect: Silent Zero-Trade Backtests & Capital Misinterpretation
- **Root Cause**: On mega-cap equities, daily predicted percentage changes are shrunken into $[-0.35\%, +0.35\%]$. Because the strategy starts in 100% cash, if forecasts never reach $+0.50\%$ (`BUY_THRESHOLD_PCT`), 0 BUY signals occur, legitimately leaving the strategy in cash. The UI lacked signal diagnostics explaining why 0 trades occurred, leading users to suspect an execution failure.
- **Resolution**:
  - Added `signal_diagnostics` to `BacktestResult` in `backtest.py`:
    - `min_pct`, `max_pct`, `mean_pct`, `std_pct`: Distribution of predicted percentage moves.
    - `buy_signals`, `sell_signals`: Counts of threshold breaches.
    - `threshold_gap`: Distance from max forecast to buy threshold.
    - `zero_trade_reason`: Clear human-readable quantitative explanation.
  - Enhanced `summary_text()` to document the exact forecast range and threshold gap when 0 trades occur.
  - Normalized `position_size_pct` so values $> 1.0$ (e.g., `100` from UI inputs) are automatically scaled to fractional `1.0` (100%), preventing inadvertent 100x leverage assumptions.
  - Enriched `renderBacktest()` in `webapp/static/ws.js` to display the quantitative diagnostic note and ledger empty state while strictly preserving all benchmark caveat assertions.

---

## 3. UI Redesign & Theming Engine

### Minimalist Quantitative Terminal with Restrained Glassmorphism
- **Design Philosophy**: High-density quantitative terminal inspired by institutional trading workstations, using system monospace and sans-serif typography (`Segoe UI`, `SF Pro`, `Cascadia Mono`) for zero CDN dependencies and strict CSP compliance.
- **Theme Palette**:
  - **Dark Theme (Default)**: Graphite surfaces (`#101215`, `#161a1f`, `#1c2127`), muted brass accent (`#c8a35a`), soft gain/loss indicators (`#5cbf8f`, `#e86f63`).
  - **Light Theme (Paper)**: Clean daylight surfaces (`#f6f8fa`, `#ffffff`, `#edf1f5`), warm amber accent (`#875d00`), forest gain (`#096e35`), crimson loss (`#cf222e`).
  - **Auto (System)**: Automatically syncs with `prefers-color-scheme` and dynamic OS switches.
- **Anti-FOUC (Flash of Unstyled Content)**: Inline script in `<head>` immediately queries `localStorage.getItem('ws.theme')` and applies `data-theme` to `<html>` prior to stylesheet and body rendering.
- **WCAG AA Compliance**: All 8 foreground text tokens on all 3 surface depths mathematically verified $\ge 4.5:1$ contrast ratio. Form control borders meet $\ge 3.0:1$ non-text contrast.
- **Glassmorphic Sticky Topbar**: `backdrop-filter: blur(12px)` with subtle border and translucent surface for seamless scrolling context.
- **Dynamic Tooltips & Cards**: Chart tooltips adapt to active theme colors with soft drop shadows; panels incorporate subtle elevation styling.

---

## 4. Cross-Platform Reliability & Test Suite Hardening

### Windows File Encoding Hardening
- **Root Cause**: On Windows systems without global `PYTHONUTF8=1`, standard `open()` in test files defaulted to `cp1252`, failing to decode Unicode glyphs (`▲`, `▼`, `◼`) in `ws.js` and `ws.css`.
- **Resolution**: Added explicit `encoding="utf-8"` to file reads in `test_v18_ui.py`, `test_v19_ui.py`, and `test_v17_webapp.py`.

### New Dedicated Regression Test Suites
- `tests/test_v21_market_data_reliability.py`:
  - `test_unknown_symbol_does_not_trip_circuit_breaker`
  - `test_tcs_bare_hint_and_ns_success`
  - `test_bare_tcs_remains_us_unspecified_by_default`
  - `test_provider_health_separates_no_data_from_outages`
- `tests/test_v21_backtest_diagnostics.py`:
  - `test_zero_trade_diagnostics_produced_for_shrunken_forecasts`
  - `test_position_size_percentage_normalized`

---

## 5. Verification & Integrity Checklist

| Check | Tool / Suite | Result | Details |
| :--- | :--- | :--- | :--- |
| Pip Consistency | `pip check` | **PASS** | No broken requirements found |
| Deep Learning Engine | TensorFlow / Keras | **PASS** | TF 2.21.0, Keras 3.15.1 import clean |
| Python Compilation | `compileall -q .` | **PASS** | All source modules compile cleanly |
| Server Startup & Smoke | `scripts/release_gate.py` | **PASS** | All 6 core routes return 200 OK |
| Market Data Reliability | `test_v21_market_data_reliability.py` | **PASS** | 4/4 passed in 1.4s |
| Backtest Diagnostics | `test_v21_backtest_diagnostics.py` | **PASS** | 2/2 passed in 1.5s |
| Static UI & Contrast | `test_v18_ui.py` | **PASS** | 7/7 passed in 0.3s |
| Charting & Whisker Isolation | `test_v19_ui.py` | **PASS** | 6/6 passed in 0.3s |
| Webapp Security & Routes | `test_v17_webapp.py` | **PASS** | 30/30 passed in 42.6s |
| Universe & View Guarantees | `test_v20_universe_and_backtest_view.py`| **PASS** | 2/2 passed in 61.3s |
