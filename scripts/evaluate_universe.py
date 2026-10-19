"""Evaluate the forecasting methodology across a LIST of instruments with real provider data -- no selection, failures included.
    python scripts/evaluate_universe.py AAPL MSFT NSE:TCS BSE:RELIANCE --epochs 15 --period 2y
Needs network access to your data provider; offline it reports every symbol as failed (it never invents data).
Writes outputs/universe_eval.json."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config, universe_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+"); ap.add_argument("--period", default="2y"); ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    out = universe_eval.evaluate_universe(a.symbols, period=a.period, epochs=a.epochs)
    print(f"{'symbol':14s} {'status':28s} {'n':>4s}  " + "  ".join(f"{m[:10]:>10s}" for m in universe_eval.MODELS))
    for r in out["rows"]:
        cells = "  ".join(f"{(r['models'].get(m, {}).get('rmse_improvement_vs_naive_pct') if r.get('models') else None) or float('nan'):>+9.1f}%" for m in universe_eval.MODELS)
        print(f"{r['symbol']:14s} {r['status']:28s} {r.get('n_obs', 0):>4d}  {cells}" + (f"   {r['error']}" if r["error"] else ""))
    print(json.dumps(out["summary"], indent=1))
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, "universe_eval.json")
    json.dump(out, open(path, "w"), indent=1); print("Saved", path)
