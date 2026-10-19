"""
trading/signal_generator.py
------------------------------
Converts the ensemble's next-day price prediction into a discrete BUY / SELL
/ HOLD signal via a simple, transparent threshold rule -- deliberately not a
black box. The thresholds are configurable in config.py.

This is a mechanical rule applied to the ensemble output, not investment
advice: it's just "if the model's predicted % change clears this bar, call
it a BUY/SELL signal, otherwise HOLD."
"""

import config


def generate_signal(final_ensemble: dict, buy_threshold: float = None,
                     sell_threshold: float = None) -> dict:
    """
    final_ensemble: the dict returned by ensemble.compute_ensemble_prediction()
    Returns: {action, predicted_pct_change, predicted_price, current_price, reasoning}
    """
    buy_threshold = config.BUY_THRESHOLD_PCT if buy_threshold is None else buy_threshold
    sell_threshold = config.SELL_THRESHOLD_PCT if sell_threshold is None else sell_threshold

    if sell_threshold >= buy_threshold:
        raise ValueError(
            f"sell_threshold ({sell_threshold}) must be less than buy_threshold "
            f"({buy_threshold}), or every prediction would be ambiguous."
        )

    pct_change = final_ensemble["predicted_pct_change"]

    # A NaN here means something upstream went wrong (a model producing
    # garbage, a division-by-zero edge case, etc.) -- it must never silently
    # fall through to HOLD, since HOLD is supposed to mean "a real
    # prediction, just not a strong enough signal", not "we don't actually
    # know". Fail loudly and specifically instead.
    if pct_change != pct_change:  # the standard NaN-detection idiom (NaN != NaN)
        raise ValueError(
            "predicted_pct_change is NaN -- this means an upstream model "
            "produced an invalid prediction. Refusing to generate a trading "
            "signal from it rather than silently treating it as HOLD."
        )

    if pct_change >= buy_threshold:
        action = "BUY"
        reasoning = (f"Ensemble predicts {pct_change:+.2f}% next-day change, "
                     f"which clears the +{buy_threshold:.2f}% BUY threshold.")
    elif pct_change <= sell_threshold:
        action = "SELL"
        reasoning = (f"Ensemble predicts {pct_change:+.2f}% next-day change, "
                     f"which clears the {sell_threshold:.2f}% SELL threshold.")
    else:
        action = "HOLD"
        reasoning = (f"Ensemble predicts {pct_change:+.2f}% next-day change, "
                     f"inside the [{sell_threshold:.2f}%, +{buy_threshold:.2f}%] "
                     f"HOLD band -- not a strong enough signal either way.")

    return {
        "action": action,
        "predicted_pct_change": pct_change,
        "predicted_price": final_ensemble["final_predicted_price"],
        "current_price": final_ensemble["current_price"],
        "reasoning": reasoning,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
    }
