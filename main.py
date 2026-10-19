"""
main.py
--------
End-to-end pipeline for the Multi-AI Stock Prediction Ensemble:

  1. Fetch historical stock data (yfinance, with Upstox/Groww fallback for
     Indian stocks)
  2. Train + evaluate a Linear Regression model
  3. Train + evaluate an LSTM model
  4. Query the LLM panel (Claude, ChatGPT, Gemini, Grok, DeepSeek) for a
     qualitative next-day forecast (any provider missing an API key is
     skipped automatically)
  5. Blend all sources into one final ensemble prediction
  6. Save comparison charts + a JSON report to outputs/

This script is pure analysis/reporting -- it never places any trades. For
the buy/sell feature, see trade.py.

Run with:  python main.py  [--ticker AAPL] [--epochs 25]
"""

import argparse
import json
import os
import sys
from datetime import datetime

import config
import pipeline
import visualize
import backtest as backtest_mod
import checkpoints as checkpoints_mod
from data_loader import InsufficientDataError
from data_sources.base import DataSourceError
from ensemble import AllModelsFailedError


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-AI Stock Prediction Ensemble")
    parser.add_argument("--ticker", type=str, default=config.TICKER)
    parser.add_argument("--period", type=str, default=config.PERIOD)
    parser.add_argument("--epochs", type=int, default=config.LSTM_EPOCHS)
    parser.add_argument("--source", type=str, default=config.DATA_SOURCE,
                         choices=["auto", "yfinance", "upstox", "groww"],
                         help="Data source: auto (default), yfinance, upstox, or groww")
    parser.add_argument("--skip-llm", action="store_true",
                         help="Skip the LLM panel entirely and just run LR + LSTM")
    parser.add_argument("--walk-forward", action="store_true",
                         help="Also run expanding-window walk-forward validation "
                              "(additional to the standard train/test split evaluation; "
                              "costs extra runtime -- retrains the LSTM a few more times)")
    parser.add_argument("--dynamic-weights", action="store_true",
                         help="Use walk-forward-validation-based ensemble weights instead "
                              "of the fixed config.ENSEMBLE_WEIGHTS (same as setting "
                              "ENSEMBLE_WEIGHTING_MODE=dynamic in .env)")
    parser.add_argument("--backtest", action="store_true",
                         help="Run the look-ahead-bias-free backtest (LR, LSTM, and "
                              "LR+LSTM ensemble) over each model's held-out test period "
                              "and compare against Buy & Hold")
    parser.add_argument("--checkpoint", type=str, default=None, metavar="NAME",
                         help="Load the LSTM from checkpoints/NAME/ if it exists and is "
                              "compatible with this ticker/config (skipping training), or "
                              "train fresh and save it under this name if it doesn't. Never "
                              "silently reuses an incompatible checkpoint -- see --list-checkpoints "
                              "and --retrain-checkpoint.")
    parser.add_argument("--retrain-checkpoint", action="store_true",
                         help="With --checkpoint NAME: force retraining even if a compatible "
                              "checkpoint already exists, then overwrite it.")
    parser.add_argument("--list-checkpoints", action="store_true",
                         help="List saved LSTM checkpoints and exit (no prediction is run).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_checkpoints:
        checkpoints_found = checkpoints_mod.list_checkpoints()
        if not checkpoints_found:
            print("No saved checkpoints found in", config.CHECKPOINT_DIR)
        else:
            print(f"Saved checkpoints ({config.CHECKPOINT_DIR}/):\n")
            for c in checkpoints_found:
                print(f"  {c['name']:20s} ticker={c['ticker']:8s} target={c['lstm_prediction_target']:6s} "
                      f"lookback={c['lookback']:3d} saved={c['saved_at']}")
        return None

    print(f"\n{'=' * 60}\nMulti-AI Stock Prediction Ensemble -- {args.ticker}\n{'=' * 60}\n")

    weighting_mode = "dynamic" if args.dynamic_weights else None
    result = pipeline.run_pipeline(
        ticker=args.ticker, period=args.period, epochs=args.epochs,
        source=args.source, skip_llm=args.skip_llm, verbose=True,
        walk_forward=args.walk_forward, ensemble_weighting_mode=weighting_mode,
        checkpoint_name=args.checkpoint, retrain=args.retrain_checkpoint,
        for_backtest=args.backtest,
    )

    backtest_results = None
    if args.backtest:
        print("\n[backtest] Running look-ahead-bias-free backtests (LR, LSTM, Ensemble)...")
        backtest_results = {}
        for model_name in ("lr", "lstm", "ensemble"):
            try:
                bt = backtest_mod.backtest_from_pipeline_result(result, model=model_name)
                backtest_results[model_name] = bt.to_dict()
                print(f"\n{bt.summary_text()}")
            except ValueError as e:
                print(f"      Skipped {model_name} backtest: {e}")

    print("\n[6/6] Saving charts and report...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    chart1 = visualize.plot_model_comparison(
        result["lr_dates_test"], result["y_test"], result["lr_eval"]["predictions"],
        result["lstm_dates_test"], result["lstm_eval"]["predictions"], ticker=args.ticker,
    )
    chart2 = visualize.plot_llm_panel_votes(result["llm_results"]) if result["llm_results"] else None
    chart3 = visualize.plot_final_ensemble(result["final_ensemble"], ticker=args.ticker)

    report = {
        "ticker": args.ticker,
        "generated_at": datetime.now().isoformat(),
        "current_price": result["current_price"],
        "data_quality": result["data_quality"],
        "run_metadata": result["run_metadata"],
        "linear_regression": {
            "next_day_prediction": result["lr_next_price"],
            "metrics": {k: v for k, v in result["lr_eval"].items() if k != "predictions"},
        },
        "lstm": {
            "next_day_prediction": result["lstm_next_price"],
            "metrics": {k: v for k, v in result["lstm_eval"].items()
                        if k not in ("predictions", "predictions_scaled", "actual",
                                     "actual_return", "predicted_return")},
        },
        "llm_panel": result["llm_panel_summary"],
        "final_ensemble": result["final_ensemble"],
        "confidence": result["confidence"],
        "walk_forward_validation": {
            "linear_regression": result["walk_forward_lr"],
            "lstm": result["walk_forward_lstm"],
        },
        "backtest": backtest_results,
        "charts": {"model_comparison": chart1, "llm_panel_votes": chart2, "final_ensemble": chart3},
    }
    report_path = os.path.join(config.OUTPUT_DIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"      Report saved to {report_path}")
    print(f"      Charts saved to {config.OUTPUT_DIR}/")
    print(f"\n{'=' * 60}\nDone.\n{'=' * 60}\n")

    return report


if __name__ == "__main__":
    try:
        main()
    except (InsufficientDataError, DataSourceError, AllModelsFailedError) as e:
        print(f"\nCould not complete the run: {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
