"""
universe_eval.py -- evaluate the SAME methodology across many instruments, with no selection.

Every symbol you pass is reported: successes, failures (data/provider/model errors) and "insufficient evaluation data". Nothing is
dropped, re-ordered or re-tried until it looks good. The summary counts how many instruments each model beat the naive
last-close baseline on out-of-sample RMSE and gives an exact two-sided SIGN TEST p-value against 50/50, so a handful of lucky
tickers cannot be mistaken for skill. Statistics across instruments of the same period are NOT independent (markets co-move);
the sign test is therefore indicative, and the summary says so.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import config
import security

MODELS = ("linear_regression", "lstm", "lr_lstm_ensemble", "drift")


def sign_test_p(wins: int, n: int) -> Optional[float]:
    """Exact two-sided binomial p-value for `wins` of `n` under p=0.5. None when n == 0."""
    if n <= 0:
        return None
    k = min(wins, n - wins)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def default_fetch(symbol: str, period: str):
    """Real data through the app's own provider layer (daily adjusted bars), partial current bar removed."""
    import pandas as pd
    import market_data, forecast_service
    svc = market_data.get_service()
    h = svc.get_history(symbol, period)
    df = pd.DataFrame(h["bars"]).rename(columns={"t": "Date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date")
    inst = market_data.resolve_instrument(symbol)
    q = svc.get_quote(symbol)
    df, _ = forecast_service.drop_partial_bar(df, inst, q.session)
    return df


def evaluate_one(symbol: str, fetch: Callable, period: str, epochs: int) -> dict:
    import data_loader, evaluation, pipeline
    row = {"symbol": symbol, "status": "failed", "error": None}
    try:
        df = fetch(symbol, period)
        data_loader.validate_sufficient_data(df)
        result = pipeline.run_pipeline(ticker=symbol, period=period, epochs=epochs, skip_llm=True, verbose=False, df=df,
                                       checkpoint_name=None)
        ev = evaluation.build_evaluation(result)
        ev.pop("_frame", None)
        row.update(status="ok" if ev["sufficient"] else "insufficient_evaluation_data", n_obs=ev["n_obs"], period=ev["period"],
                   n_dropped_nonfinite=ev["n_dropped_nonfinite"], n_dropped_gap=ev["n_dropped_gap"], epochs=epochs,
                   models={m["model"]: {k: m.get(k) for k in ("rmse", "mae", "mase", "directional_accuracy_pct", "rmse_improvement_vs_naive_pct",
                                                       "rmse_improvement_ci95", "rmse_improvement_ci99")}
                           | {"dm": m.get("diebold_mariano_vs_naive")} for m in ev["models"] if m["model"] != "naive"},
                   naive_rmse=next(m["rmse"] for m in ev["models"] if m["model"] == "naive"), error=None)
    except Exception as e:  # noqa: BLE001 -- recorded, never hidden
        row["error"] = security.redact(f"{type(e).__name__}: {e}")[:300]
    return row


def summarize(rows: List[dict]) -> dict:
    usable = [r for r in rows if r["status"] == "ok"]
    per: Dict[str, dict] = {}
    for m in MODELS:
        imps = [r["models"][m]["rmse_improvement_vs_naive_pct"] for r in usable
                if r["models"].get(m, {}).get("rmse_improvement_vs_naive_pct") is not None]
        wins = sum(i > 0 for i in imps)
        s = sorted(imps)
        per[m] = {"n_instruments": len(imps), "n_beat_naive": wins, "n_not_beat_naive": len(imps) - wins,
                  "median_rmse_improvement_pct": s[len(s) // 2] if s else None,
                  "worst_pct": s[0] if s else None, "best_pct": s[-1] if s else None,
                  "n_significantly_better": sum(1 for r in usable if (r["models"].get(m, {}).get("rmse_improvement_ci99") or [0, 0])[0] > 0),
                  "n_significantly_worse": sum(1 for r in usable if (r["models"].get(m, {}).get("rmse_improvement_ci99") or [0, 0])[1] < 0),
                  "sign_test_p_two_sided": sign_test_p(wins, len(imps))}
    return {"requested": len(rows), "evaluated_ok": len(usable),
            "failed": sum(r["status"] == "failed" for r in rows),
            "insufficient_evaluation_data": sum(r["status"] == "insufficient_evaluation_data" for r in rows),
            "per_model": per,
            "reading_guide": ("All requested instruments are listed, including failures. 'beat naive' = lower out-of-sample RMSE than the "
                              "last-close forecast on that instrument's own held-out days; 'significantly better/worse' = the 99% block-bootstrap interval excludes 0. Instruments are not independent, and "
                              "beating naive on RMSE is not a trading edge."),
            }


def evaluate_universe(symbols: List[str], *, period: str = "2y", epochs: int = None, fetch: Callable = None) -> dict:
    epochs = int(epochs or config.LSTM_EPOCHS)
    fetch = fetch or default_fetch
    rows = [evaluate_one(s, fetch, period, epochs) for s in symbols]            # order and membership exactly as requested
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "period": period, "epochs": epochs,
            "symbols_requested": list(symbols), "rows": rows, "summary": summarize(rows)}
