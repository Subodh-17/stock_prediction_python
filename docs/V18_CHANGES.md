# V18 change record

Classification used throughout: **PASS** = executed and passed · **FAIL** = executed and failed · **BLOCKED** = could not be
executed (no network / credentials / platform) · **NOT RUN** = not attempted. A mocked HTTP success is *not* a provider test.

## 1. Verification matrix (what is actually known)

| Item | Result | Evidence / reason |
|---|---|---|
| Python compilation (`compileall`) | **PASS** | V17 baseline and V18 tree |
| Original (pre-V17) tests, 188, unmodified | **PASS** | byte-identical to the V16 zip; see section 6 |
| Full suite run **from the extracted ZIP**, in a venv created from that ZIP's `requirements-dev-lock.txt` | **PASS** | **380 passed, 0 failed, 0 skipped** in 267 s (188 original + 110 V17 + 82 V18), Python 3.12.3, Linux x86_64 (an earlier run of the same code from the work tree, in a venv built from the runtime lock + pytest, gave the same 380/0 in 236 s) |
| Clean install from the delivered lock | **PASS** (Linux x86_64, CPython 3.12.3) | second clean venv -> identical 80-package set; `pip check` clean |
| Fresh start with no `.env`, no keys | **PASS** | app serves `/`, all AI providers report `missing_key` |
| Install on **Windows** / macOS / other Pythons | **NOT RUN** | no such platform available. Legacy Windows 3.11 lock kept, not re-run |
| Real Yahoo history / quote request | **BLOCKED** | host unreachable from the build sandbox |
| Real Upstox quote request | **BLOCKED** | host unreachable; also no token |
| Real Groww request | **BLOCKED** | host unreachable; no credentials |
| Real Gemini / Groq / OpenRouter / OpenAI / Anthropic / xAI / DeepSeek request | **BLOCKED** | hosts unreachable; no API keys |
| Real-browser audit (Chromium, 1440/820/390 px, all 10 routes) | **PASS** | 0 console/CSP errors, 0 overflow, focus visible, **axe-core WCAG 2.2 A/AA: 0 violations on 30 pages** -- run against the QA harness (synthetic inputs, fake AI HTTP), so it audits the UI, not provider behaviour |
| Official documentation consulted | done | Google Gemini deprecations + pricing pages; Upstox market-quote docs (see section 4) |

**No provider is verified.** Until you run `python scripts/verify_providers.py` with your own credentials (section 8), every provider
integration is "implemented and mock-tested only", and the app's Data & Model Health page says so.

## 2. Issues found (verified against the V17 source) and fixed

| # | Issue | Where | Fix |
|---|---|---|---|
| 1 | **Conformal interval mismatch**: coverage was measured with the *first-half* quantile but the displayed interval used a quantile over *all* residuals incl. the holdout; rank was silently clamped to the sample max | `evaluation.conformal_interval` | Replaced by strictly-past rolling calibration (`rolling_conformal`, `forecast_interval`); sample too small -> explicit "Insufficient evaluation data" |
| 2 | Checkpoints recorded no training start/end, so a reused model could have seen the "out-of-sample" window (e.g. after the dataset shrank) | `checkpoints.py`, `pipeline.py` | Provenance manifest v2, data fingerprints, integrity hashes, strict cutoff (section 3.2) |
| 3 | **Yahoo broken on a fresh install**: importing TensorFlow before yfinance's HTTP client makes every request fail locally with `ImpersonateError: Impersonating chrome150 is not supported` | env (yfinance 1.7.0 + curl_cffi 0.16.3 + TF 2.21) | `_native_preload.py` runs first from `config`; actionable `client_init` error; subprocess regression test |
| 4 | Upstox adapter could never parse `last_trade_time` (docs: epoch **milliseconds as a string**) -> always "unverified" | `market_data._parse_ts` | Parser handles it; tested with the documented example value |
| 5 | Legacy resolver served **NSE** data for `.BO` (BSE) tickers via Upstox/Groww and, in auto mode, a different listing after a Yahoo failure | `data_sources/__init__.py` | Explicit `exchange="BSE"` for `.BO`; suffix-fallback resolution is recorded and shown on the results page |
| 6 | UI showed "null / null" for train/test rows, lookback, split, seed (wrong metadata keys) | `forecast_service.py` | Correct keys + new end-to-end test asserting real values |
| 7 | Race in `submit_forecast` (check and register in separate lock acquisitions) -> duplicate jobs possible | `forecast_service.py` | One atomic section (RLock); 8-thread test |
| 8 | `/predict` and `/api/predict` ran the whole pipeline inside the web request | `webapp/app.py` | Background jobs (section 3.4) |
| 9 | Job/API error text could include file paths; V17 loggers bypassed the project's redaction (tracebacks) | `forecast_service.py`, `webapp/api_v2.py` | Whitelisted user-safe error types; loggers moved to `stockpred.*` |
| 10 | Text contrast: `--dim` 3.3-3.8:1 and control borders 1.7:1 (WCAG AA needs 4.5:1 / 3:1) | `ws.css` | Colours computed to pass; regression test parses the stylesheet |
| 11 | Scrollable tables not keyboard-reachable (axe `scrollable-region-focusable`) | `ws.js` | focusable, named regions |
| 12 | Charts had no non-visual equivalent | `ws.js` | `<details>` data table for every chart |
| 13 | Old lock's note called `httpx2` "not a real package" (it is -- `anthropic` requires it); lock claimed it was the tested set after dependencies changed | `requirements-lock.txt` | Regenerated; old file kept as `requirements-lock-windows-py311-legacy.txt` |
| 14 | A test double could be displayed as a verified provider | `ai_integrations/providers.py` | `real_transport` flag: injected transport -> `mock_verified` |
| 15 | `.env.example` / README described Windows 3.11 lock as *the* tested set | docs | Rewritten |

## 3. Methodology and rules

### 3.1 Prediction intervals (`evaluation.py`)
Roles (all chronological): **TRAIN** = rows the LR/LSTM were fitted on (never used for calibration or scoring). **RESIDUAL** = one
absolute return residual per out-of-sample date, each from a model that never saw that day. **CALIBRATION** for the interval of
target date *t* = the most recent `window` (120) residuals dated strictly **before** *t* (the residual of *t-1* is known once *t-1*
closes). **EVALUATION** = every date with at least 40 earlier residuals; its coverage is measured against **the very intervals
that were produced** (the interval displayed for a past day *is* the one scored). A **new forecast** uses the identical procedure
on the latest residuals, all dated before the forecast target. Quantile rank = `ceil((m+1)*level)`; if it exceeds `m` the interval
is unavailable (never clamped). Needs >= 30 evaluated intervals, else "Insufficient evaluation data". Non-finite residuals are
dropped **and counted**.
Recorded: calibration window/period/count, quantile method, nominal coverage (`nominal_coverage`), observed coverage
(`observed_coverage`, a separate field), mean width, assessment (under/over/close). **Limitation:** conformal guarantees assume
exchangeable residuals; returns are serially dependent and non-stationary, so **no coverage guarantee is claimed**. A rolling window
adapts to slow regime change, not to sudden jumps (a test shows coverage collapsing after a volatility jump, and the UI reports it).
The coverage shown beside a *forecast* interval is the historical coverage of the **procedure**; it does not validate that interval.

### 3.2 Checkpoint provenance (`checkpoints.py`)
Every checkpoint written by V18 stores (in `metadata.json`, `manifest_version: 2`): model type, implementation version, ticker,
exchange, horizon, lookback, feature names + feature-engineering version, **train_start / train_end**, a SHA-256 fingerprint of the
training rows and of the last 60 training rows, scaler config, training config, creation time, seed, package versions, and SHA-256 of
`model.keras` and `scaler.json`. A checkpoint is reused only if **all** hold: ticker, lookback, target, split, horizon, feature schema,
exchange, model-implementation and feature versions match; the manifest exists; hashes match; `train_end` is **strictly before the
first evaluated date** (so it never saw the evaluation window / simulated future); `train_end` exists in the current data and the
last 60 training rows still match (catches restated/dividend-adjusted history). Anything else raises `CheckpointMismatchError` (or its
subclass `CheckpointProvenanceError`) and the pipeline retrains. **Missing manifest** (V17 and earlier): documented migration = retrain
and overwrite *that name only*. Saving is atomic (temp dir + replace) and touches only its own directory; a rejected checkpoint is
never deleted, and other checkpoints are never modified. (Integrity hashes are checked before a model is *returned*, after the semantic
checks so a swapped file gets the most specific diagnosis; Keras `safe_mode` still governs deserialisation.)

### 3.3 Evaluation and backtest (what is implemented vs limitations)
Implemented: same out-of-sample dates for naive / drift / LR / LSTM / LR+LSTM; MAE, RMSE, return-RMSE, MAPE (undefined near zero),
directional accuracy (flat days excluded), improvement vs naive, approximate Diebold-Mariano z; period and `n_obs` beside every metric;
non-finite rows dropped and counted; < 30 days -> "Insufficient evaluation data" with no verdict. Weights: the evaluated ensemble is fixed
(50/50 default) -- **nothing is tuned on the final holdout**. Legacy `dynamic` weighting was read and is leakage-aware in backtests (each
day uses only walk-forward folds that ended strictly before that day). Scalers are fitted on training data only (original tests).
Backtest: signal for day *i* uses data through *i-1*; default fill = day *i* **open**; missing/invalid open = **no fill**; costs per
side + slippage bps; mark-to-market at close; Buy & Hold uses the same entry rule/costs. **Limitations:** long-only single instrument;
adjusted prices (dividends embedded, taxes ignored; history can be restated by the provider); no partial fills, spreads, market-session or
liquidity modelling; a historical backtest of the *final* model on days it was out-of-sample is valid, but thresholds tuned on the same
window are in-sample. Unit tests establish correct mechanics, **not** that any model has real-world predictive skill.

### 3.4 Background jobs
`forecast_service.submit_task` runs work on one worker; identical in-flight requests share a job; status endpoint
`GET /api/v2/jobs/<id>`; cooperative cancel (between stages only); a running job past `JOB_TIMEOUT_SECONDS` is reported `timeout`; a
failed/cancelled job never sets a result and **clears** the session's previous result id. **Compatibility trade-off (stated plainly):** the
HTML forms send `async=1` and always get an immediate `202` + a bounded-polling page. Clients that do **not** send `async` (scripts, and
the original tests, which run the real pipeline and expect a synchronous result) wait up to `LEGACY_PREDICT_WAIT_SECONDS` (default 120 s)
and then also get `202`. That wait still occupies a request thread; send `async=1` / `"async": true` to avoid it. `/api/signal` and
`/api/backtest` remain synchronous (rate limited, read-only compute).

## 4. Provider reference (endpoints, entitlements, status)

| Provider | Endpoint class / auth | Coverage & intervals | Delay semantics | Limits / retry | Verified |
|---|---|---|---|---|---|
| Yahoo via `yfinance` 1.7.0 | unofficial Yahoo endpoints, no key | most exchanges; daily bars in this app (history), quote via `info` or last 1-minute bar | never classified `live`; `delayed` only when Yahoo reports `exchangeDataDelayedBy`; otherwise `unverified` | unofficial, personal-use; app backs off with jitter, honours `Retry-After`, opens a circuit after 3 failures | **BLOCKED** (+ import-order bug fixed) |
| Upstox quote | `GET https://api.upstox.com/v2/market-quote/quotes?instrument_key=…`, Bearer token (daily) | NSE/BSE; "snapshots obtained directly from the exchanges" (official docs); up to 500 keys | `last_trade_time` = epoch **ms string**; `live` only if account entitled **and** timestamp <= `LIVE_MAX_AGE_SECONDS`. An illiquid instrument's last trade can be older -> shown `stale`/`unverified`, never `live` | per Upstox; 429 `Retry-After` honoured | **BLOCKED** |
| Upstox history / Groww history | existing adapters (daily) | NSE (default) / **BSE now passed explicitly for `.BO`** | EOD / historical | per provider | **BLOCKED** |
| Google Gemini | `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`, header `x-goog-api-key`; `GET …/models` lists models | default `gemini-3.5-flash-lite`; **`gemini-2.0-flash` was shut down 2026-06-01** (official deprecations page); 2.5 models limited to accounts that used them; model availability is per account | n/a | free tier exists for several Flash/Flash-Lite models; limits are per project -- read AI Studio. On the free tier Google may use prompts to improve products | **BLOCKED** |
| Groq / OpenRouter | OpenAI-compatible `…/chat/completions`, Bearer; `GET …/models` | **no default model** (ids are account-specific) | n/a | free-tier terms **not researched** | **BLOCKED** |
| OpenAI / Anthropic / xAI / DeepSeek | their public APIs | paid; a chat subscription does not include API credit | n/a | per provider | **BLOCKED** |

Gemini facts come from Google's official pages as fetched on the build date; they can change. Upstox facts from its official
market-quote documentation.

## 5. AI integration status
Error classes: `missing_key, invalid_key, model_unavailable, unsupported_endpoint, rate_limit, insufficient_quota, network, timeout,
invalid_response, provider_outage, bad_request`. A provider is **`verified` only after a real request through the real transport
succeeds**; an injected transport (tests, the QA harness) can only reach `mock_verified`. HTTP 200 with an empty/blocked/thought-only
body is `invalid_response`. Failures start a cooldown so refreshes do not hammer a provider. LLM numeric weight is **0 by default** and
the LLM is not evaluated numerically (no point-in-time historical LLM forecasts exist; generating them now would use hindsight).
Commentary is separated from the quantitative forecast, validated against a fact pack, and numbers not in the facts are flagged.
**No AI model was verified** (BLOCKED).

## 6. Tests
Original tests (188) are unmodified. **V17 tests changed in V18:** two interval tests in `tests/test_v17_evaluation.py` were removed
(they asserted the old mixed-quantile behaviour; replaced by `tests/test_v18_conformal.py`), and one assertion in
`tests/test_v17_ai.py` now expects `mock_verified` for an injected transport. New V18 files: `test_v18_checkpoints`, `test_v18_conformal`,
`test_v18_jobs`, `test_v18_providers`, `test_v18_forecast_e2e`, `test_v18_native_preload`, `test_v18_security`, `test_v18_ui`.
Result in the lock-built environment: **380 passed, 0 failed, 0 skipped** in 267 s (188 original + 110 V17 + 82 V18), Python 3.12.3, Linux x86_64 (an earlier run of the same code from the work tree, in a venv built from the runtime lock + pytest, gave the same 380/0 in 236 s). Mutation checks (leaking the evaluated residual into calibration; breaking the
next-open fill; removing the real-time entitlement check) each made the relevant tests fail.
A passing test establishes the mechanics, not real-world prediction accuracy or provider correctness.

## 7. Dependencies
Regenerate the lock: `python3.12 -m venv v && v/bin/pip install -r requirements.txt -r webapp/requirements.txt && v/bin/pip freeze`
(drop pip/setuptools/wheel), then install that file into a second clean venv and compare `pip freeze`. Supported: CPython 3.12, Linux
x86_64. The legacy Windows/3.11.0 lock is retained unchanged; it was user-run for V16 and **has not been re-run against V18**.

## 8. How to verify genuine market-data and AI access (run on your machine)
```powershell
# put your keys in .env (never commit it), then:
python scripts\verify_providers.py
```
It makes **real, read-only** requests (history, quotes, an LLM test prompt; never an order) and writes `outputs/provider_verification.json`
(provider, endpoint class, instrument, UTC time, outcome; no secrets). Outcomes: `PASS` (real response validated: price > 0, source
timestamp present, symbol/exchange match, OHLC consistent), `FAIL`, `BLOCKED` (host unreachable), `NOT_CONFIGURED`. A `PASS` for a quote
still reports its **freshness category** -- Yahoo will not say `live`. The **Data & Model Health** page shows these records; if the file
is absent it says nothing has been verified.

## 9. Free-tier AI
Create a key in Google AI Studio -> `.env`: `GOOGLE_API_KEY=...` -> restart -> **AI Research -> Test connection**. It lists the models your
key can use; set `GOOGLE_MODEL` to one (free-tier eligibility is per account/project; check Google's current pricing and rate-limit pages).
The app stays fully usable with no AI key.

## 10. Windows PowerShell
```powershell
cd stock_prediction_python
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r webapp/requirements.txt
Copy-Item .env.example .env
python scripts\verify_providers.py          # optional: genuine provider checks with your keys
python webapp\app.py                         # http://127.0.0.1:5000  (binds to localhost only)
```
(Tests: `pip install -r requirements-dev.txt; python -m pytest tests webapp/test_app.py -q`.) The Windows path is **unverified for V18**.

## 11. Running safely
Paper trading is the default and the only mode the workstation API can reach. Live trading needs `ENABLE_LIVE_TRADING=true`, a 32+ character
`WEBAPP_LIVE_TRADING_TOKEN`, the operator secret and an explicit confirmation step in the classic flow; no `/api/v2` route can place an order.
The app binds to `127.0.0.1`; do not expose it publicly. CSRF is enforced on every state-changing route; JSON responses are `no-store`.

## 12. Known limitations / unresolved
* Nothing was tested against a live provider (section 1). Windows/macOS installs unverified.
* `ImpersonateError` mechanism is a hypothesis (native-library load order); behaviour was reproduced deterministically and the fix verified in the lock environment only.
* Programmatic `/predict` without `async` still holds a request thread up to 120 s (compatibility with the original tests).
* Cancellation does not interrupt a running Keras fit.
* axe-core finds only a subset of accessibility issues; no screen-reader testing was done; contrast was verified numerically for text/control colours, not for canvas chart colours.
* Evaluation/backtest results on real markets were **not** produced here (no market data); only fixture-based mechanics tests.
* The Upstox quote adapter and every AI adapter are unverified live.
