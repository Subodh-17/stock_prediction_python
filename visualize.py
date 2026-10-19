"""
visualize.py
-------------
Plots for comparing Linear Regression vs LSTM performance on the test set,
and a bar chart summarizing the LLM panel's votes. Saves PNGs to the
outputs/ directory.
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

import config


def plot_model_comparison(lr_dates_test, lr_y_true, lr_preds,
                           lstm_dates_test, lstm_preds, save_path=None, ticker: str = None):
    """
    LR and LSTM use different windowing schemes (LR uses simple lagged
    features, LSTM uses a `lookback`-day sequence), so their test sets don't
    start on exactly the same date and can differ in length by a few rows.
    Both series are plotted against their own date axis on the same chart --
    the "Actual Close" line (from the LR test set, which is a superset of
    the LSTM dates) is the shared reference both prediction lines are judged
    against.

    `ticker` should be the actual runtime-selected ticker (see the same note
    in data_loader.recent_trend_summary -- this function used to fall back
    silently to config.TICKER, which could mislabel the chart title for
    whatever ticker was actually run). Falls back to config.TICKER only for
    backward-compatible standalone calls that don't pass it.
    """
    ticker_label = ticker if ticker else config.TICKER
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lr_dates_test, lr_y_true, label="Actual Close", color="black", linewidth=1.8)
    ax.plot(lr_dates_test, lr_preds, label="Linear Regression", color="tab:blue",
            linestyle="--", linewidth=1.4)
    ax.plot(lstm_dates_test, lstm_preds, label="LSTM", color="tab:red",
            linestyle="--", linewidth=1.4)
    ax.set_title(f"{ticker_label}: Actual vs Predicted Closing Price (Test Set)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    save_path = save_path or os.path.join(config.OUTPUT_DIR, "model_comparison.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_llm_panel_votes(llm_results: list, save_path=None):
    valid = [r for r in llm_results if r.get("error") is None]
    if not valid:
        return None

    providers = [r["provider"] for r in valid]
    pct_changes = [r["estimated_pct_change"] for r in valid]
    confidences = [r["confidence"] for r in valid]
    colors = ["tab:green" if p >= 0 else "tab:red" for p in pct_changes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(providers, pct_changes, color=colors)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("LLM Panel: Estimated Next-Day % Change")
    ax1.set_ylabel("% change")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(providers, confidences, color="tab:purple", alpha=0.7)
    ax2.set_title("LLM Panel: Self-Reported Confidence")
    ax2.set_ylabel("confidence (0-1)")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    save_path = save_path or os.path.join(config.OUTPUT_DIR, "llm_panel_votes.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_final_ensemble(ensemble_result: dict, save_path=None, ticker: str = None):
    ticker_label = ticker if ticker else config.TICKER
    components = ensemble_result["component_predictions"]
    labels = list(components.keys()) + ["Final Ensemble"]
    values = list(components.values()) + [ensemble_result["final_predicted_price"]]
    colors = ["tab:blue", "tab:red", "tab:orange", "black"][:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(ensemble_result["current_price"], color="gray", linestyle=":",
               label=f"Current price (${ensemble_result['current_price']:.2f})")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"${v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{ticker_label}: Next-Day Price Prediction by Source")
    ax.set_ylabel("Stock Price")
    ax.legend()
    fig.tight_layout()

    save_path = save_path or os.path.join(config.OUTPUT_DIR, "final_ensemble.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
