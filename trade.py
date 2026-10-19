"""
trade.py
---------
Runs the full prediction pipeline (LR + LSTM + LLM panel + ensemble, same as
main.py) and then acts on it: generates a BUY/SELL/HOLD signal and, unless
told otherwise, executes it against the PAPER (simulated) broker.

SAFE BY DEFAULT: every run is paper trading unless you pass --live AND have
set ENABLE_LIVE_TRADING=true in your .env. See README.md's "Trading" section
before ever using --live.

Examples:
    python trade.py --ticker AAPL                       # predict + paper trade
    python trade.py --ticker AAPL --skip-llm             # faster, LR+LSTM only
    python trade.py --ticker RELIANCE --source upstox --broker upstox --live
                                                           # REAL order via Upstox
                                                           # (requires ENABLE_LIVE_TRADING=true in .env)
"""

import argparse
import sys

import config
import pipeline
from trading.trade_executor import execute_trade_decision
from data_loader import InsufficientDataError
from data_sources.base import DataSourceError
from ensemble import AllModelsFailedError


def parse_args():
    parser = argparse.ArgumentParser(description="Predict, then buy/sell based on the ensemble signal")
    parser.add_argument("--ticker", type=str, default=config.TICKER)
    parser.add_argument("--period", type=str, default=config.PERIOD)
    parser.add_argument("--epochs", type=int, default=config.LSTM_EPOCHS)
    parser.add_argument("--source", type=str, default=config.DATA_SOURCE,
                         choices=["auto", "yfinance", "upstox", "groww"])
    parser.add_argument("--skip-llm", action="store_true",
                         help="Skip the LLM panel and just use LR + LSTM for the signal")
    parser.add_argument("--mode", type=str, default="paper", choices=["paper", "live"],
                         help="paper (default, simulated) or live (real orders -- needs "
                              "ENABLE_LIVE_TRADING=true in .env too)")
    parser.add_argument("--broker", type=str, default=None, choices=["paper", "upstox", "groww"],
                         help="Which broker to use. Defaults to config.DEFAULT_BROKER "
                              "(paper) for paper mode, or must be upstox/groww for live mode.")
    parser.add_argument("--capital", type=float, default=None,
                         help="Override available capital for position sizing "
                              "(otherwise fetched from the broker, or PAPER_STARTING_CASH for paper mode)")
    parser.add_argument("--quantity", type=int, default=None,
                         help="Override the computed order quantity and trade this exact "
                              "number of shares instead (still subject to risk_manager validation)")
    parser.add_argument("--auto-confirm", action="store_true",
                         help="Skip the interactive confirmation prompt for live orders. "
                              "Only use this in reviewed, unattended automation -- it removes "
                              "the last human checkpoint before a real order is submitted.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "live":
        print("\n*** LIVE TRADING MODE REQUESTED ***")
        if not config.ENABLE_LIVE_TRADING:
            print("ENABLE_LIVE_TRADING is not set to 'true' in your .env file.")
            print("Live trading stays disabled until you edit .env yourself -- this is intentional.")
            return
        if args.broker not in ("upstox", "groww"):
            print("Live mode requires --broker upstox or --broker groww.")
            return

    print(f"\n{'=' * 60}\nPredicting {args.ticker}...\n{'=' * 60}\n")
    result = pipeline.run_pipeline(
        ticker=args.ticker, period=args.period, epochs=args.epochs,
        source=args.source, skip_llm=args.skip_llm, verbose=True,
    )

    print(f"\n{'=' * 60}\nGenerating trade decision...\n{'=' * 60}\n")
    outcome = execute_trade_decision(
        ticker=args.ticker,
        final_ensemble=result["final_ensemble"],
        mode=args.mode,
        broker_name=args.broker,
        capital_override=args.capital,
        quantity_override=args.quantity,
        auto_confirm=args.auto_confirm,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print(f"Executed: {outcome['executed']}")
    print(f"Reason:   {outcome['reason']}")
    print(f"{'=' * 60}\n")
    return outcome


if __name__ == "__main__":
    try:
        main()
    except (InsufficientDataError, DataSourceError, AllModelsFailedError) as e:
        print(f"\nCould not complete the run: {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
