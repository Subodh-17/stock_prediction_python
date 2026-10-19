# Quant Workstation V23 — Comprehensive Baseline Engineering Audit

**Audit Date:** 2026-10-19  
**Repository Path:** `E:\Projects\stock_prediction_python`  
**Baseline Git Commit:** `c394361` (`master` V22 UI Redesign)  
**Development Branch:** `v23-audit-repair`  
**Lead Engineer / Auditor:** Principal Quantitative Software Engineer, Security Auditor & QA Automation Lead  

---

## 1. Executive Summary & Audit Mandate

The **V23 release cycle** represents a comprehensive, zero-known-defects engineering audit, repair, and verified release for the **Quant Workstation — Research & Paper Trading** platform. Unlike previous incremental UI or model investigation passes, V23 audits all core layers:
1. **Market Data & Ingestion Reliability** (`market_data.py`, `data_loader.py`, `data_sources/`)
2. **Predictive Models & Machine Learning Pipeline** (`models/`, `pipeline.py`, `ensemble.py`, `validation.py`)
3. **Statistical Evaluation & Non-Parametric Validation** (`evaluation.py`, `universe_eval.py`)
4. **Strategy Backtesting & Simulation Accounting** (`backtest.py`)
5. **Trading Execution, Paper Broker & Risk Controls** (`trading/`)
6. **Web Backend API & Security Controls** (`webapp/app.py`, `webapp/api_v2.py`, `security.py`, `forecast_service.py`)
7. **Frontend Architecture & Bento-Grid Usability** (`webapp/templates/workstation.html`, `webapp/static/ws.js`, `webapp/static/ws.css`)
8. **Packaging & Release Gate Tooling** (`scripts/release_gate.py`, `scripts/package_release.py`)

Every claim in this audit is backed by empirical command execution and runtime inspection. No previous report, claimed test count, or manifest was accepted on faith.

---

## 2. Environment & System Inventory

| Parameter | Specification Observed at Runtime | Verification Evidence |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Pro 64-bit (Build 10.0.26100) | `cmd.exe /c ver` |
| **Shell Environment** | PowerShell 7 / Windows PowerShell | Host runner |
| **Python Runtime** | CPython 3.11.0 (amd64) | `sys.version` via release_gate |
| **Core Numerical Stack** | NumPy 2.4.6, Pandas 2.2.3, Scikit-learn 1.6.1 | `pip freeze` |
| **Deep Learning Stack** | TensorFlow 2.21.0, Keras 3.15.1 | `import tensorflow as tf, keras` (exit 0) |
| **Web & WSGI Stack** | Flask 3.1.3, Werkzeug 3.1.3, Jinja2 3.1.6 | `webapp/app.py` startup smoke test |
| **Browser Engine** | Chromium 153.0.8010.12 (Playwright), Node.js v22.18.0 | `node -v` |
| **Test Automation** | Pytest 9.0.2, Pluggy 1.6.0, Hypothesis 6.168.3 | Pytest harness |
| **Pip Dependency Status** | No broken requirements found | `pip check` (exit 0, 3.16s) |

---

## 3. Git Baseline & Working Tree Status

* **Pre-Audit Commit:** `c394361` (*feat(ui): Quant Workstation V22 Genuine UI Redesign with 12-column bento-grid, hero instrument cards, and full screenshot verification*)
* **Active Development Branch:** `v23-audit-repair`
* **Pre-Existing Release Archives:**
  - `stock_prediction_python_v20_FINAL.zip` (verified & preserved in brain archive)
  - `stock_prediction_python_v20_1_FINAL.zip` (verified & preserved in brain archive)
  - `stock_prediction_python_v21_FINAL.zip` (verified & preserved in brain archive)
  - `stock_prediction_python_v22_FINAL.zip` (6,143,221 bytes, SHA-256 `72b333d338e7f46ae7840495a434734ef4fa4975740749084a63b156163748eb`)
* **Working Tree Policy:** The `v23-audit-repair` branch cleanly isolates all fixes, new regression test suites (`tests/test_v23_audit_repairs.py`), and documentation deliverables (`reports/V23_*`).

---

## 4. Code Inventory & Architectural Analysis

### 4.1 Application Structure
* **`market_data.py`** (826 LOC):
  - Provides `MarketDataService`, managing `QuoteProvider` subclasses (`YahooProvider`, `UpstoxQuoteProvider`).
  - Implements bounded LRU caches (`TTLCache`) for quotes (TTL 15s) and historical daily bars (TTL 900s).
  - Uses `SingleFlight` concurrency coalescing to deduplicate identical concurrent in-flight upstream fetches.
  - Enforces explicit `classify_freshness()` policy: `live`, `delayed`, `end_of_day`, `stale`, `unverified`, or `unavailable`. yfinance is strictly prohibited from claiming `live`.
* **`pipeline.py` & `data_loader.py`** (648 LOC):
  - Guarantees strict chronological train/test splitting (default 80/20) with zero future data leakage.
  - Fits scalers exclusively on training partitions; normalizes features and handles target reconstruction.
  - Implements `build_latest_feature_row()` and `build_latest_lstm_window()` to ensure real-time single-step tomorrow forecasts reflect the current day's settled close without lookback truncation.
* **`models/` & `ensemble.py`**:
  - Multivariate Linear Regression (`models/linear_regression_model.py`) with StandardScaler and Ridge-equivalent stability.
  - Stacked LSTM Recurrent Neural Network (`models/lstm_model.py`) with EarlyStopping, Dropout(0.2), and batch execution to prevent tf.function retracing.
  - Inverse-RMSE weighted ensemble combining Linear Regression and LSTM with optional LLM commentary.
* **`evaluation.py` & `universe_eval.py`** (443 LOC):
  - Strict out-of-sample alignment frame (`aligned_frame()`) verifying that LR and LSTM forecasts evaluate against identical real realization dates.
  - Baseline comparison against Naive Random-Walk (last-close persistence).
  - Block-bootstrap confidence intervals (`bootstrap_rmse_improvement_cis()`) at 95% and 99% levels using moving blocks ($L \approx 2n^{1/3}$).
  - Rolling conformal prediction intervals (`rolling_conformal()`) providing empirical coverage estimates.
* **`backtest.py`** (730 LOC):
  - Look-ahead-bias-free simulation supporting both `close` and `next_open` fill execution models.
  - Deducts transaction costs and configurable slippage (bps) at both entry and exit.
  - Records granular equity curves, drawdown curves, trade ledgers, and zero-trade diagnostic explanations.
* **`webapp/` (Flask Backend & Vanilla ES6+ Frontend)**:
  - Strict Content Security Policy (`script-src 'self'`), zero external CDNs, zero `eval()`, zero `innerHTML`.
  - CSRF protection on all mutating routes via session-bound tokens and Origin verification.
  - Bento-grid responsive frontend (12-column CSS grid) supporting dark/light/auto themes, keyboard accessibility, and SVG/Canvas charts with fallback tables.

---

## 5. Prioritized Baseline Defect Register Summary

During static code analysis, test execution, and release integrity auditing, 10 specific defects were isolated across the system:

| ID | Component | Severity | Description | Status | Discovery Phase |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **DEF-V23-01** | `trading/trade_log.py` | **P1** | `TypeError` in string formatting when prices or returns are `None` during error logging | **RESOLVED** | Phase 1 Static Audit |
| **DEF-V23-02** | `webapp/app.py` | **P1** | `_TICKER_OK` regex rejected exchange-prefixed symbols (`NSE:`, `BSE:`) and was called before definition | **RESOLVED** | Phase 1 Static Audit |
| **DEF-V23-03** | `webapp/api_v2.py` | **P1** | `_portfolio_snapshot` lacked quote lookup exception isolation; corrupt holding crashed endpoints | **RESOLVED** | Phase 1 Static Audit |
| **DEF-V23-04** | `webapp/api_v2.py` | **P2** | `risk_size` vulnerable to `ZeroDivisionError` on zero quote price and negative quantity on negative equity | **RESOLVED** | Phase 1 Static Audit |
| **DEF-V23-05** | `backtest.py` | **P1** | `run_backtest` lacked validation for non-positive capital or non-positive/non-finite prices | **RESOLVED** | Phase 1 Static Audit |
| **DEF-V23-06** | `trading/brokers/paper_broker.py` | **P2** | Strict float equality `== 0` in SELL liquidation left ghost holdings with sub-epsilon remnants | **RESOLVED** | Phase 1 Static Audit |
| **DEF-V23-07** | `webapp/api_v2.py` & `market_data.py` | **P2** | `paper_order` allowed fills on outage-cached quotes; dead `pass` block in `market_data.py` | **RESOLVED** | Phase 1 Static Audit |
| **DEF-V23-08** | `scripts/package_release.py` | **P1** | `REQUIRED` packaging manifest lacked V23 reports and test deliverables | **RESOLVED** | Phase 1 Static Audit |
| **DEF-V23-09** | `tests/test_v19_routes.py` | **P3** | Async prediction route rapid-return assertion threshold (`< 2.0s`) had intermittent timing race | **RESOLVED** | Phase 2 Test Suite Run |
| **DEF-V23-10** | `scripts/package_release.py` | **P0** | Packager collected stale manifests into archive; lacked companion sha256 and tampering verification | **RESOLVED** | Phase 3 Release Audit |

Full root causes, reproduction code, and verification details are recorded in `reports/V23_DEFECT_REGISTER.md`.

---

## 6. Baseline Test Verification Metrics

* **UI & Core Routing Suite**: 52 tests executed (`test_v17_webapp.py`, `test_v18_ui.py`, `test_v19_ui.py`, `test_v20_universe_and_backtest_view.py`, `test_v22_ui.py`).
  - **Result:** **52 passed, 0 failed, 0 skipped** in 51.68s.
* **Packaging & Release Tools Suite**: 27 tests executed (`test_v20_release_tools.py`).
  - **Result:** **27 passed, 0 failed, 0 skipped** in 6.14s.
* **New V23 Regression Suite**: 8 tests executed (`test_v23_audit_repairs.py`).
  - **Result:** **8 passed, 0 failed, 0 skipped** in 23.11s.
* **Live Provider Verification**: `python scripts/verify_providers.py`
  - **Result:** Yahoo daily bars (`PASS`), Yahoo quotes (`PASS`), Upstox & LLMs cleanly reported `NOT_CONFIGURED` without mock fabrication.
