# V20 Final Verification Report

Final Release Gate Status: **PASS WITH LIMITATIONS**

Every verification step that could be executed was independently run and verified against genuine endpoints, test suites, and models. All 545 tests collected and passed with 0 failures, 0 skips, and 0 errors. Real market data provider integration (Yahoo Finance) was verified live over the network with real daily bars and quotes. Live credentials for Upstox and AI providers were not provided in the environment and are documented truthfully as `NOT_CONFIGURED`. Real-market model evaluation on a fixed 4-ticker universe confirmed that machine learning models do not statistically outperform the naive last-close baseline out of sample.

---

## 1. Source ZIP and Manifest Verification

The source archive and manifest files provided for verification were validated before extraction:

| Attribute | Manifest Expectation | Actual Calculated Value | Match |
|---|---|---|---|
| Archive Filename | `stock_prediction_python_v20.zip` | `stock_prediction_python_v20.zip` | MATCH |
| Archive Byte Size | 439,453 bytes | 439,453 bytes | MATCH |
| File Count | 131 files | 131 files | MATCH |
| SHA-256 Hash | `79c58ac90a3ce65658f6cf3ffd814e62808c8a61ec88e20a99f201cab94503bc` | `79c58ac90a3ce65658f6cf3ffd814e62808c8a61ec88e20a99f201cab94503bc` | MATCH |
| Duplicate Paths | 0 | 0 | MATCH |
| Archive Integrity (`testzip`) | Clean (`None`) | Clean (`None`) | MATCH |
| Required Files Present | 18 | 18 | MATCH |

The original archive and manifest have been preserved untouched in `original/`.

---

## 2. Before / After Change Summary

### Defects Identified and Repaired

1. **Cross-Platform Test Execution and Encoding Defect (Windows `cp1252`)**:
   - **Finding**: On Windows hosts where `PYTHONUTF8` is not set by default, test collection crashed with `UnicodeDecodeError: 'charmap'` when loading static assets (`webapp/static/ws.js`) containing UTF-8 glyphs (such as `●` at byte 3910).
   - **Fix**:
     - Updated `scripts/release_gate.py` to establish `os.environ.setdefault("PYTHONUTF8", "1")` and explicitly pass `PYTHONUTF8: "1"` to subprocess environments for `run_suite` and `smoke_test_server`.
     - Updated `tests/test_v20_universe_and_backtest_view.py` line 11 to explicitly specify `encoding="utf-8"` when reading `ws.js`.
     - Added `--lock` argument to `scripts/release_gate.py` allowing explicit selection of platform-specific lock files while preserving the default behavior.

2. **Dangling Package Freeze Crash**:
   - **Finding**: Running `pip freeze` in the host environment failed with `NotADirectoryError: [WinError 267]` because of a dangling editable package link referencing a removed directory, causing `release_gate.py` to observe an empty installed package set.
   - **Fix**: Resolved the missing path in the development environment, restoring normal `pip freeze` operation.

3. **Pinned Dev Static Analysis Dependency Missing**:
   - **Finding**: Test `test_v19_static_analysis.py::test_no_undefined_names_anywhere_in_the_repository` was skipped because `pyflakes` was missing from the host environment.
   - **Fix**: Installed pinned `pyflakes==4.0.3` (from `requirements-dev-lock.txt`). The test ran and passed cleanly.

4. **NumPy and ML-Dtypes Incompatibility in Host Environment**:
   - **Finding**: `pip check` initially failed due to an incompatible NumPy version (1.26.4) with `ml-dtypes 0.6.0`.
   - **Fix**: Aligned NumPy to `numpy==2.4.6` (as pinned in `requirements-lock-windows-py311-legacy.txt`). `pip check` returned 0 ("No broken requirements found.").

---

## 3. Test File Preservation and Modifications

- **Pre-existing test files**: All 34 pre-existing test files and test helpers from V19 were preserved byte-for-byte. None were skipped, weakened, or rewritten.
- **Modified test files**: 1 V20 test file (`tests/test_v20_universe_and_backtest_view.py`) was updated at line 11 to include `encoding="utf-8"` to prevent Windows locale encoding errors.
- **Removed test files**: None.
- **Added test files**: None (all tests reside in the 7 verified `test_v20_*` files).

---

## 4. Test Suite Execution Results

The full test suite was executed through the release gate runner:

```bash
python scripts/release_gate.py --tests-only
```

- **Exit Code**: 0
- **Elapsed Time**: 659.70s (pytest reported runtime: 626.39s)
- **Collected**: 545
- **Passed**: 545
- **Failed**: 0
- **Skipped**: 0
- **Errors**: 0

All 103 new V20 tests passed:
- `tests/test_v20_backtest_accounting.py`: 12 passed
- `tests/test_v20_cli_and_news.py`: 15 passed
- `tests/test_v20_data_integrity.py`: 7 passed
- `tests/test_v20_evaluation_stats.py`: 10 passed
- `tests/test_v20_release_tools.py`: 27 passed
- `tests/test_v20_security.py`: 30 passed
- `tests/test_v20_universe_and_backtest_view.py`: 2 passed

---

## 5. Dependency, TensorFlow, and Environment Verification

| Check | Command | Exit Code | Elapsed | Result / Evidence |
|---|---|---|---|---|
| Python Version | `python --version` | 0 | 0.0s | CPython 3.11.0 (Windows AMD64) |
| Dependency Integrity | `python -m pip check` | 0 | 3.1s | "No broken requirements found." |
| Real TensorFlow Import | `python -c "import tensorflow as tf, keras; ..."` | 0 | 6.5s | `tensorflow 2.21.0 keras 3.15.1` |
| Source Compilation | `python -m compileall -q .` | 0 | 0.2s | All modules compiled cleanly |
| Server Startup Smoke Test | `smoke_test_server` | 0 | 7.3s | 6 endpoints returned HTTP 200 |

### Dependency Lock Consistency Analysis
- **Linux Reference (`requirements-dev-lock.txt` / `requirements-lock.txt`)**: Pinned against CPython 3.12 / Linux x86_64. 0 required packages missing. On Windows / Python 3.11, platform-specific differences exist (e.g. `scipy 1.18.1` requires Python >=3.12, whereas Windows / Python 3.11 uses `scipy 1.17.1`).
- **Windows Reference (`requirements-lock-windows-py311-legacy.txt`)**: All core runtime dependencies (`numpy 2.4.6`, `pandas 3.0.5`, `scikit-learn 1.9.1`, `tensorflow-cpu 2.21.0`, `keras 3.15.1`, `yfinance 1.7.0`, `Flask 3.1.3`, `reportlab 5.0.1`, `pyflakes 4.0.3`) match the verified Windows execution environment.

---

## 6. Real Market Data Provider Verification

Real network requests were executed using `scripts/verify_providers.py` (recorded to `outputs/provider_verification.json`):

| Provider | Endpoint Class | Target Instrument | Outcome | Verification Evidence |
|---|---|---|---|---|
| Yahoo Finance | History (daily bars) | `AAPL` | **PASS** | 22 daily bars retrieved (`2026-09-17` to `2026-10-16`), adjusted=True, valid OHLC boundaries, 0 data quality issues. |
| Yahoo Finance | Quote | `AAPL` | **PASS** | Real price retrieved, exchange `US/UNSPECIFIED`, currency `USD`, freshness `stale` (weekend capture), flags: `['older_than_expected_delay']`. |
| Upstox | Quote | `NSE:RELIANCE` | **NOT_CONFIGURED** | Daily access token / credentials not set in `.env`. |
| Groww | History / Quote | - | **NOT_CONFIGURED** | Credentials not configured in local environment. |

Synthetic fallback tests verify that provider failures cannot silently inject synthetic data into user workflows.

---

## 7. AI Provider Integrations

All AI provider client adapters were inspected and tested via `scripts/verify_providers.py`:

| Provider | Model / Endpoint | Status | Finding |
|---|---|---|---|
| Gemini | `gemini-3.5-flash-lite` | **NOT_CONFIGURED** | `GOOGLE_API_KEY` not set |
| Groq | `llama-3.3-70b-versatile` | **NOT_CONFIGURED** | `GROQ_API_KEY` not set |
| OpenRouter | Auto / Configured | **NOT_CONFIGURED** | `OPENROUTER_API_KEY` not set |
| Anthropic | `claude-3-5-haiku` | **NOT_CONFIGURED** | `ANTHROPIC_API_KEY` not set |
| OpenAI | `gpt-4o-mini` | **NOT_CONFIGURED** | `OPENAI_API_KEY` not set |
| xAI | `grok-2` | **NOT_CONFIGURED** | `XAI_API_KEY` not set |
| DeepSeek | `deepseek-chat` | **NOT_CONFIGURED** | `DEEPSEEK_API_KEY` not set |

All offline error-handling, bounded retry, timeout, rate-limit, and secret-redaction mechanisms passed in the regression test suite (`tests/test_v17_ai.py`, `tests/test_v18_providers.py`). Numeric forecasts remain strictly separated from AI commentary and are never influenced by missing AI keys.

---

## 8. Real-Market Model Evaluation

A reproducible evaluation was performed on a pre-declared universe of 4 major assets (`AAPL`, `MSFT`, `GOOGL`, `AMZN`) over a 2-year historical period using `scripts/evaluate_universe.py` (saved to `outputs/universe_eval.json`).

### Out-of-Sample Performance Comparison (99 Evaluation Days per Ticker)

Evaluation period: **2026-05-28 to 2026-10-16** (1-day forward forecast horizon, next-day close).

#### AAPL (Naive RMSE: 5.534)
- **Linear Regression**: RMSE = 5.767, MAE = 4.215, MASE = 1.528, Dir. Acc. = 55.6%, RMSE vs Naive = -4.21% (99% CI: [-12.29%, +3.58%], DM z = -0.934, p = 0.350)
- **LSTM**: RMSE = 5.809, MAE = 4.166, MASE = 1.510, Dir. Acc. = 54.5%, RMSE vs Naive = -4.95% (99% CI: [-10.71%, +3.22%], DM z = -1.453, p = 0.146)
- **LR+LSTM Ensemble**: RMSE = 5.560, MAE = 3.994, MASE = 1.448, Dir. Acc. = 53.5%, RMSE vs Naive = -0.46% (99% CI: [-6.08%, +4.63%], DM z = -0.223, p = 0.824)
- **Drift Baseline**: RMSE = 5.527, MAE = 3.963, MASE = 1.437, Dir. Acc. = 54.5%, RMSE vs Naive = +0.14% (99% CI: [-0.91%, +1.73%], DM z = +0.269, p = 0.788)

#### MSFT (Naive RMSE: 8.895)
- **Linear Regression**: RMSE = 8.973, MAE = 6.643, MASE = 1.488, Dir. Acc. = 47.5%, RMSE vs Naive = -0.88% (99% CI: [-4.63%, +1.74%], DM z = -0.730, p = 0.465)
- **LSTM**: RMSE = 8.973, MAE = 6.690, MASE = 1.499, Dir. Acc. = 53.5%, RMSE vs Naive = -0.88% (99% CI: [-4.00%, +1.64%], DM z = -0.738, p = 0.461)
- **LR+LSTM Ensemble**: RMSE = 8.850, MAE = 6.541, MASE = 1.465, Dir. Acc. = 52.5%, RMSE vs Naive = +0.50% (99% CI: [-1.43%, +1.98%], DM z = +0.672, p = 0.502)
- **Drift Baseline**: RMSE = 8.889, MAE = 6.582, MASE = 1.474, Dir. Acc. = 52.5%, RMSE vs Naive = +0.06% (99% CI: [-0.34%, +0.47%], DM z = +0.457, p = 0.648)

#### GOOGL (Naive RMSE: 3.824)
- **Linear Regression**: RMSE = 3.883, MAE = 2.916, MASE = 1.486, Dir. Acc. = 45.5%, RMSE vs Naive = -1.54% (99% CI: [-7.02%, +2.53%], DM z = -0.638, p = 0.523)
- **LSTM**: RMSE = 3.974, MAE = 3.031, MASE = 1.545, Dir. Acc. = 51.5%, RMSE vs Naive = -3.93% (99% CI: [-8.42%, -0.21%], DM z = -2.167, p = 0.030)
- **LR+LSTM Ensemble**: RMSE = 3.921, MAE = 2.973, MASE = 1.515, Dir. Acc. = 48.5%, RMSE vs Naive = -2.54% (99% CI: [-6.40%, +0.66%], DM z = -1.527, p = 0.127)
- **Drift Baseline**: RMSE = 3.884, MAE = 2.966, MASE = 1.511, Dir. Acc. = 46.5%, RMSE vs Naive = -1.58% (99% CI: [-3.49%, -0.06%], DM z = -2.316, p = 0.021)

#### AMZN (Naive RMSE: 4.887)
- **Linear Regression**: RMSE = 4.835, MAE = 3.738, MASE = 1.408, Dir. Acc. = 52.5%, RMSE vs Naive = +1.07% (99% CI: [-2.14%, +4.12%], DM z = +0.728, p = 0.467)
- **LSTM**: RMSE = 5.099, MAE = 3.987, MASE = 1.502, Dir. Acc. = 51.5%, RMSE vs Naive = -4.34% (99% CI: [-8.35%, -0.47%], DM z = -2.483, p = 0.013)
- **LR+LSTM Ensemble**: RMSE = 4.903, MAE = 3.811, MASE = 1.436, Dir. Acc. = 52.5%, RMSE vs Naive = -0.32% (99% CI: [-3.68%, +2.48%], DM z = -0.301, p = 0.763)
- **Drift Baseline**: RMSE = 4.907, MAE = 3.805, MASE = 1.433, Dir. Acc. = 51.5%, RMSE vs Naive = -0.40% (99% CI: [-1.43%, +0.33%], DM z = -1.134, p = 0.257)

### Statistical Verdict
- **Instruments significantly outperforming naive at 99% bootstrap confidence**: **0 of 4** for all models.
- **Instruments significantly underperforming naive at 99% confidence**: LSTM was statistically worse on 2 of 4 assets (`GOOGL`, `AMZN`).
- **Conclusion**: Neither Linear Regression, LSTM, nor their Ensemble demonstrates statistically significant skill over the naive random-walk baseline out-of-sample on real market data. The system truthfully reports this verdict in accordance with strict release gate criteria.

---

## 9. Backtesting and Uncertainty Findings

1. **Trade P&L Accounting (Defect D1)**:
   - Verified that the entry fee is factored directly into the trade's cost basis (`cost_basis = shares * entry_price + entry_fee`).
   - Verified that round-trip P&L reconciles exactly with cumulative portfolio equity changes across all cost configurations.
2. **Next-Open Execution**:
   - Causally valid next-open execution mode correctly fills at day `i`'s open based solely on signals generated through day `i-1`.
   - Missing open prices correctly trigger `skipped_fills` without guessing.
3. **Slippage and Sizing**:
   - Sizing constraints, cash availability checks, adverse slippage (bps), and turnover ratio (`turnover_x`) function correctly.
4. **Prediction Intervals**:
   - Conformal prediction intervals on return residuals operate strictly past-only without lookahead leakage.

---

## 10. Security Audit Findings

1. **Input Validation (Defect D2)**:
   - Strict ticker validation enforced across all routes (1-20 characters, `A-Z 0-9 . - & =`, optional leading `^`).
   - Injection payloads (`<script>`, SQL fragments, path traversals `../../etc/passwd`, null bytes, control characters) return HTTP 400.
2. **Model Deserialization**:
   - Keras models are explicitly loaded with `safe_mode=True`.
   - Checkpoints containing Python `Lambda` layers or corrupt weights fail verification safely without executing arbitrary bytecode.
3. **Credential & Secret Protection**:
   - The packager enforces regex-based scanning for private keys, AWS tokens, API keys, and certificates.
   - `.env.example` contains placeholders only. Real secrets are never logged or exposed in responses.

---

## 11. Known Limitations and Checks Not Run

1. **Live Provider Verification**:
   - `NOT RUN — credentials unavailable` for Upstox and all AI LLM providers (Gemini, Groq, OpenRouter, Anthropic, OpenAI, xAI, DeepSeek). Real network connectivity was confirmed, but authentication requires user-provided secrets.
2. **Trading Calendar (Risk R2)**:
   - Holiday schedules and exchange half-days are based on heuristic market session close tables rather than full exchange calendar libraries. The system conservatively drops open session bars.
3. **Forecast Horizon (Limitation R6)**:
   - Current architecture models a single-day (1-bar) forward horizon only.
4. **Production Server Deployment (Risk R5)**:
   - The embedded web server runs the Flask development server on `127.0.0.1`. A WSGI server (such as Waitress or Gunicorn) and authentication reverse proxy must be deployed prior to exposing the application beyond localhost.

---

## 12. Final Release Artifact Details

- **Archive Path**: `C:\Users\Subodh\.gemini\antigravity\scratch\v20_release\stock_prediction_python_v20_FINAL.zip`
- **Manifest Path**: `C:\Users\Subodh\.gemini\antigravity\scratch\v20_release\stock_prediction_python_v20_FINAL.zip.manifest.json`
- **File Count**: 132 files
- **Required Files Count**: 19
- **Integrity**: Clean (`testzip` returned `None`; 0 duplicate paths; 0 corrupt members)
- **Verification**: Self-verified via `scripts/package_release.py` and accompanying manifest JSON.
