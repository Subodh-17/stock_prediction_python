"""Measured latency of the app's own stages, on SYNTHETIC fixture bars (clearly NOT real-market latency: provider network time is
not included; a fake provider with a fixed 300 ms delay stands in for the upstream).
    python scripts/benchmark.py [--epochs 5] [--bars 520]
Prints a table and writes outputs/benchmark.json. Every number is a wall-clock measurement taken by this run."""
import argparse, json, os, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--bars", type=int, default=520); ap.add_argument("--warmup", action="store_true", help="call forecast_service.warm_up() first (as the web server does at start)")
a = ap.parse_args()
tmp = tempfile.mkdtemp(prefix="bench_")
os.environ.update(CHECKPOINT_DIR=tmp + "/ck", PAPER_PORTFOLIO_PATH=tmp + "/p.json", TRADE_LOG_PATH=tmp + "/l.csv", OUTPUT_DIR=tmp + "/o")
import config, perf, market_data as md, forecast_service as fs
from tests.v17_helpers import fixture_bars
from tests.test_v17_market_data import Fake, raw
import data_sources.yfinance_source as ys

bars = fixture_bars(a.bars, seed=21, start="2023-01-02")
ys.fetch = lambda *x, **k: bars.copy()
prov = Fake("yahoo", raw=raw(price=100.0), delay=0.30)       # fixed 300 ms stand-in for upstream latency
svc = md.MarketDataService([prov]); md._service = svc
if a.warmup:
    _w = fs.warm_up()
out = {"warmup": (_w if a.warmup else None), "disclaimer": "synthetic fixture bars; fake provider with a fixed 300 ms delay; CPU of the machine that ran this", "bars": a.bars, "epochs": a.epochs}

def wait(j):
    while j["status"] not in ("done", "failed", "cancelled"): time.sleep(0.05)
    assert j["status"] == "done", j
t = time.perf_counter(); svc.get_quote("AAPL", force=True); out["quote_miss_ms"] = round((time.perf_counter() - t) * 1000, 1)
hits = []
for _ in range(500):
    t = time.perf_counter(); svc.get_quote("AAPL"); hits.append((time.perf_counter() - t) * 1000)
hits.sort(); out["quote_cache_hit_ms"] = {"median": round(hits[250], 4), "p95": round(hits[475], 4), "upstream_calls_for_501_requests": prov.calls}
t = time.perf_counter(); j = fs.submit_forecast("AAPL", "2y", a.epochs, svc=svc); out["submit_returns_ms"] = round((time.perf_counter() - t) * 1000, 1)
t = time.perf_counter(); wait(j); out["forecast_train_total_s"] = round(time.perf_counter() - t, 2); out["forecast_stage_timings_ms"] = dict(j["timings"])
t = time.perf_counter(); j2 = fs.submit_forecast("AAPL", "2y", a.epochs, svc=svc); out["repeat_same_bar_ms"] = round((time.perf_counter() - t) * 1000, 2); out["repeat_was_cache_hit"] = bool(j2["timings"].get("cached"))
fs._RESULTS.clear()
t = time.perf_counter(); j3 = fs.submit_forecast("AAPL", "2y", a.epochs, svc=svc); wait(j3); out["forecast_checkpoint_reuse_total_s"] = round(time.perf_counter() - t, 2)
bt = []
for _ in range(20):
    t = time.perf_counter(); fs.run_backtest_job(j3["result_key"], {"model": "lr_lstm_ensemble", "buy_threshold_pct": 0.05, "sell_threshold_pct": -0.05}); bt.append((time.perf_counter() - t) * 1000)
bt.sort(); out["backtest_ms"] = {"median": round(bt[10], 1), "p95": round(bt[19], 1)}
out["measured_by_the_app"] = perf.snapshot()
print(json.dumps(out, indent=1))
os.makedirs("outputs", exist_ok=True); json.dump(out, open("outputs/benchmark.json", "w"), indent=1)
