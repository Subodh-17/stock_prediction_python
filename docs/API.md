# JSON API reference

Base URL when running locally: `http://127.0.0.1:5000`

All `POST` endpoints accept **either** a JSON body (`Content-Type:
application/json`) **or** form-encoded data. Unknown fields are ignored
rather than rejected, so adding a field to a client never breaks it.

Responses are always JSON. No endpoint ever returns a Python traceback.

---

## Authentication

The prediction and backtest endpoints are **unauthenticated** — they are
read-only analysis over public market data and carry the same trust model as
the CLI (bind to `127.0.0.1`, single user).

**No API endpoint can place a trade of any kind.** Trading is reachable only
through the HTML routes (`/trade`, `/trade/confirm`, `/trade/execute`), and
live trading there additionally requires `ENABLE_LIVE_TRADING=true` plus the
`WEBAPP_LIVE_TRADING_TOKEN` operator secret. See the README's "Safety model".

---

## `GET /api/health`

Liveness check. Takes no input and returns no configuration or secrets.

```json
{ "status": "ok", "service": "stock-prediction-ensemble" }
```

`200` always, if the process is up.

---

## `POST /api/predict`

Runs the full pipeline and returns the prediction, metrics and signal.

| Field | Type | Required | Default | Validation |
|---|---|---|---|---|
| `ticker` | string | yes | — | non-empty, ≤20 chars, alphanumeric plus `.` and `-` |
| `period` | string | no | `PERIOD` | one of `1mo 3mo 6mo 1y 2y 5y 10y ytd max` |
| `epochs` | int | no | `LSTM_EPOCHS` | 1–500 |
| `source` | string | no | `DATA_SOURCE` | one of `auto yfinance upstox groww` |
| `skip_llm` | bool-ish | no | `false` | `1/true/on/yes` |
| `walk_forward` | bool-ish | no | `false` | as above |
| `dynamic_weights` | bool-ish | no | `false` | as above |
| `for_backtest` | bool-ish | no | `false` | as above |

Response (`200`):

```jsonc
{
  "ticker": "AAPL",
  "current_price": 227.51,
  "data_quality": { "ticker": "...", "n_rows": 320, "issues": [], "warnings": [], "is_valid": true },
  "linear_regression": {
    "next_day_prediction": 227.18,
    "metrics": { "rmse": 3.024, "mae": 2.4, "r2": 0.97, "mape": 1.02,
                 "accuracy_pct": 39.3, "directional_accuracy_pct": 39.3 }
  },
  "lstm":              { "next_day_prediction": 228.40, "metrics": { /* same shape */ } },
  "llm_panel":         null,        // or the blended panel summary
  "final_ensemble": {
    "final_predicted_price": 227.81,
    "current_price": 227.51,
    "predicted_pct_change": 0.13,
    "component_predictions": { "linear_regression": 227.18, "lstm": 228.40 },
    "weights_used": { "linear_regression": 0.44, "lstm": 0.56, "llm_panel": 0.0, "mode": "dynamic" },
    "excluded_components": []       // any model whose prediction was unusable
  },
  "confidence":   { "confidence_label": "Moderate", /* ... */ },
  "signal":       { "action": "HOLD", "predicted_pct_change": 0.13, /* ... */ },
  "walk_forward": { "linear_regression": null, "lstm": null },
  "run_metadata": { "random_seed": 42, "package_versions": { /* ... */ }, /* ... */ }
}
```

**Metric naming.** `directional_accuracy_pct` is the percentage of held-out
days whose predicted *direction* was correct. `accuracy_pct` is the same
number under its legacy key, kept for backward compatibility — new consumers
should read `directional_accuracy_pct`. It is **not** classification accuracy
and says nothing about price-forecast error (that is `rmse`/`mae`/`mape`).
All model metrics are computed on the **held-out test split**, never on
training data.

---

## `POST /api/signal`

Fast path: skips the LLM panel and walk-forward validation.

Accepts `ticker` (required), `period`, `epochs` — same validation as above.
Returns `ticker`, `signal`, `final_ensemble`, `confidence`.

---

## `POST /api/backtest`

Runs the pipeline, then a **simulated** backtest over the held-out period.

| Field | Type | Required | Default | Validation |
|---|---|---|---|---|
| `ticker` | string | yes | — | as above |
| `period`, `epochs` | | no | | as above |
| `model` | string | no | `ensemble` | one of `lr lstm ensemble` |
| `weighting_mode` | string | no | `auto` | one of `auto fixed dynamic` |
| `dynamic_weights` | bool-ish | no | `false` | |

Returns the serialized `BacktestResult`: `total_return_pct`,
`buy_hold_return_pct`, `max_drawdown_pct`, `sharpe_ratio`, `sortino_ratio`,
`calmar_ratio`, `num_trades`, `win_rate_pct`, `profit_factor`,
`avg_trade_return_pct`, `total_transaction_costs`, `trades`, `equity_curve`,
`assumptions`, `weighting_mode`, and the exact thresholds/costs used.

Ratio fields are `null` rather than infinity when undefined (no losing
trades, no drawdown, no downside deviation).

**These are simulated results over a historical period under the assumptions
stated in `assumptions`. They are not evidence of profitability and not a
forecast.**

---

## Error schema

Every error has the same shape and an appropriate status code:

```json
{ "error": "invalid_period", "message": "'period' must be one of [...]." }
```

| Status | `error` | Meaning |
|---|---|---|
| `400` | `invalid_ticker`, `invalid_period`, `invalid_epochs`, `invalid_source`, `invalid_model`, `invalid_weighting_mode` | Input failed validation. Malformed JSON also returns `400`. |
| `413` | — | Request body exceeded `WEBAPP_MAX_CONTENT_LENGTH_BYTES`. |
| `422` | `insufficient_data` | Not enough history for a usable train/test split. |
| `422` | `all_models_failed` | No component model produced a usable prediction. |
| `422` | `backtest_error` | Not enough overlapping data, or `weighting_mode=dynamic` without walk-forward fold data. |
| `502` | `data_source_error` | Upstream market-data failure. Bodies are redacted and truncated. |
| `500` | `internal_error` | Deliberately generic. Details go to the server log, never the response. |

No error response contains an API key, a broker credential, an environment
variable, a filesystem path, or a stack trace.

---

## Determinism

With the same input data, configuration and `RANDOM_SEED`, `/api/predict` is
reproducible — measured bit-identical for LR, LSTM and the ensemble on CPU in
the tested environment. This is not a formal guarantee; see the README's
"Reproducibility" section for the exact limits.


---

# V17 workstation API (`/api/v2`)

JSON, same-origin. `GET` routes are read-only and never return secrets. Every `POST` requires the header
`X-CSRF-Token` (fetch it from `GET /api/v2/meta`) and, if an `Origin` header is sent, it must match the host.
Expensive routes are rate limited per IP (`WEBAPP_EXPENSIVE_RPM`, default 12/min) and return `429` + `Retry-After`.
Errors: `{"error": "<code>", "message": "<redacted, bounded text>"}`. No route can place a live order.

| Method | Path | Purpose |
|---|---|---|
| GET | `/meta` | API version, CSRF token, trading mode (always paper here) |
| GET | `/quote?symbol=&force=1` | One quote with provenance + freshness (see below) |
| GET | `/quotes?symbols=A,B` | Up to 12 quotes (cached, de-duplicated) |
| GET | `/history?symbol=&period=` | Adjusted daily OHLCV, data-quality report, indicators |
| GET | `/news?symbol=` | Headlines from `NEWS_FEEDS`, or `status: not_configured` |
| POST | `/forecast` | `{symbol, period, epochs, walk_forward}` -> `202` + job (background) |
| GET | `/jobs/<id>`, POST `/jobs/<id>/cancel` | Job status; cooperative cancel |
| GET | `/forecast/result?key=` | Forecast + evaluation payload |
| POST | `/backtest` | `{key, model, initial_capital, transaction_cost_pct, slippage_bps, buy/sell_threshold_pct, position_size_pct, execution, start, end}` |
| GET | `/ai/status` | Per-provider state (`verified` only after a real request succeeded) |
| POST | `/ai/test` | `{provider}` -- real connection test; lists models the key can use |
| POST | `/ai/research` | `{key, force}` -- validated, fact-grounded commentary |
| GET | `/portfolio`, `/risk`, `/risk/size` | Paper portfolio, exposure, position-size calculator |
| POST | `/paper/order`, `/paper/reset` | Simulated order (needs `confirm:true` + usable quote); reset balance |
| GET | `/health`, `/settings` | Measured provider/cache/job health; non-secret settings |

## Quote object
`provider, symbol, exchange, currency, price, open, high, low, prev_close, volume, source_ts, retrieved_ts, timezone,
session, data_type, delay_seconds, freshness, age_seconds, is_stale, realtime_eligible, from_cache, cache_age_seconds,
quality_flags, selection_trail, downgraded_from, latency_ms, error`.

`freshness` is one of `live | delayed | end_of_day | stale | unverified | unavailable` (`historical` for bar series).
`live` requires a provider with a real-time entitlement **and** a source timestamp within `LIVE_MAX_AGE_SECONDS`.
`retrieved_ts` never implies freshness. `selection_trail` lists every provider tried; `downgraded_from` is set whenever a
lower-freshness fallback answered.


---

# V18 API changes

## Classic `/predict` and `/api/predict` are now background jobs (bounded wait)
The pipeline no longer runs inside the web request. The route submits a job and waits at most
`LEGACY_PREDICT_WAIT_SECONDS` (default 2 s):
* finished in time -> exactly the old behaviour (`/predict`: redirect to `/results`; `/api/predict`: `200` + JSON);
* otherwise -> **`202`**. `/predict` returns a page that polls `GET /api/v2/jobs/<id>` with a bounded schedule and then
  loads `GET /predict/claim/<id>`; `/api/predict` returns `{status:"running", job_id, status_url, result_url}`.
  Poll `GET /api/predict/result/<job_id>` (`202` while running, `200` + the usual JSON when done, or the original error
  status/code if it failed).
* Identical in-flight requests share one job. A failed or cancelled job never sets `result_id`, and it clears any previous
  one, so an older forecast is never shown as the result of a newer request.
* `GET /api/v2/jobs/<id>` -> `{id, kind, status: queued|running|done|failed|cancelled|timeout, stage, elapsed_seconds,
  timed_out?, error?, result_key?}`; a running job older than `JOB_TIMEOUT_SECONDS` is reported as `timeout`.
`/api/signal` and `/api/backtest` remain synchronous (rate limited, CSRF-exempt, read-only compute; they cannot place orders).

## Prediction interval object (`forecast.interval`)
```
status: "ok" | "insufficient_evaluation_data"
level / nominal_coverage          the requested coverage (NOT an observed value)
low, high, q_return               the interval for the next, unobserved day
calibration {n, start, end}       the latest residuals used; all dated before the forecast target
evaluation  {observed_coverage, n_evaluated, n_covered, evaluation_period, mean_relative_width, assessment, window, ...}
                                  historical coverage of THE SAME PROCEDURE; evidence about the method, not a guarantee
```
`evaluation.interval_evaluation.series[]` lists every historical interval that was scored `{date, low, high, point, actual,
covered, q_return, n_calibration}` -- the interval shown for a past day *is* the interval that was evaluated.

## Other additions
* `GET /api/v2/health` -> `verification.records[]` (written only by `scripts/verify_providers.py`; empty = nothing verified).
* `GET /api/v2/ai/status` -> each provider has `state` in `missing_key | untested | verified | mock_verified | <error class>`
  and `transport` (`real` | `test double`). `verified` is impossible for a test double.
* Every `/api/v2` response is `Cache-Control: no-store`.


---

# V19 API changes
* `POST /api/signal` and `POST /api/backtest` are background jobs like `/api/predict`: same `async` opt-in (`"async": true` -> immediate
  `202`), bounded synchronous wait otherwise. Poll `GET /api/jobs/<job_id>/result` (`202` while running; `200` + the usual JSON when done;
  the original error status/code if it failed). `GET /api/predict/result/<id>` still works. Job ids are now 24 hex characters.
* `GET /api/v2/health` adds `performance` (measured latencies by operation; empty until something has run) and
  `market_data.history_failure_cache_hits`.
* `GET /api/v2/history` / the history layer: a failed fetch is remembered for `HISTORY_FAILURE_TTL_SECONDS` and re-raised without
  contacting the provider (the error says so).
* Symbols: Yahoo index symbols with a leading `^` are accepted (`^GSPC`, `^NSEI`, `^BSESN`, ...) with an explicit exchange for common ones.
* Evaluation payload adds `n_dropped_gap` and `max_gap_days`: targets whose base bar is more than 5 calendar days earlier are excluded.
* Classic results payload adds `dropped_open_session_bar` (a bar for a session still in progress is removed before modelling).
* `GET /api/v2/settings` adds `models.llm_weight_note` (why a requested LLM weight was ignored).


---

# V20 API changes
* **Evaluation payload** (`forecast.evaluation`): every non-naive model row adds `mase`, `rmse_improvement_ci95`, `rmse_improvement_ci99`
  (moving-block bootstrap, percent) and `directional_accuracy_ci95`; the payload adds `horizon` (`days: 1` and its definition),
  `mase_note` and `ci_note`. `conclusion` says *beats* only when the **99%** interval's lower bound is above 0; otherwise it says *does NOT
  beat* and whether the model is *WORSE* or merely *not distinguishable*. Metrics that cannot be computed are `null` with a reason in `unavailable`.
* **Backtest result** adds `exposure_pct` (share of simulated days holding a position at the close) and `turnover_x` (total traded value / initial
  capital). **Behaviour change:** each trade's `cost_basis` and `pnl` now include the entry fee, so `sum(pnl)` of closed trades equals the change in
  equity; `win_rate_pct`, `profit_factor` and `avg_trade_return_pct` are computed from those corrected values.
* **Tickers:** the classic `POST /predict` form now applies the same rule as the JSON API: 1-20 characters, ASCII letters/digits and `. - & =`,
  optionally starting with `^`. Anything else is `400`. (`&`, `=` and a leading `^` are newly accepted on the JSON API too.)
* **Data quality** (`history.data_quality`): new warnings for >= 5 consecutive days of identical OHLC (possible frozen feed), >= 5 consecutive
  zero-volume days on instruments that otherwise report volume, and a timezone-aware index; infinite values are now reported by name. Timezone-aware
  provider indexes are normalised to exchange-local dates.
