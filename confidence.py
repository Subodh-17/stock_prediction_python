"""
confidence.py
----------------
A lightweight confidence/uncertainty layer on top of the ensemble's final
prediction.

IMPORTANT, stated up front: nothing in this module is a formally valid
statistical confidence interval or a calibrated probability. Building one
properly would require modeling the actual residual DISTRIBUTION (e.g. via
quantile regression, conformal prediction, or a Bayesian model) and
validating its coverage against held-out data -- none of which this
project's models do. What follows is a transparent, documented HEURISTIC:
it combines (1) how much the component models agree with each other and
(2) each model's own historical error magnitude into a labeled summary. It
is useful for calibrating how much weight a person should put on a given
prediction relative to another one from this same system, but it should
never be read as "there is an X% statistical chance the price falls in this
range."
"""

from typing import Optional

import numpy as np


def compute_model_agreement(component_predictions: dict, current_price: float) -> dict:
    """
    Quantifies how much the ensemble's component models agree with each
    other, in both DIRECTION (up/down/flat relative to current_price) and
    MAGNITUDE (the spread between their predicted % changes).

    `component_predictions`: e.g. ensemble_result["component_predictions"],
    a dict of {model_name: predicted_price}. Works with 2 or 3 components
    (LR+LSTM, or LR+LSTM+LLM panel).
    """
    if not component_predictions:
        return {
            "direction_agreement": None, "pct_change_spread": None,
            "agreement_label": "Unknown", "component_pct_changes": {},
            "component_directions": {},
        }

    pct_changes = {}
    directions = {}
    for name, price in component_predictions.items():
        pct = (price - current_price) / current_price * 100
        pct_changes[name] = pct
        # A small dead-zone around 0% avoids labeling two near-identical
        # predictions as "disagreeing" over a fraction of a percent.
        if pct > 0.1:
            directions[name] = "up"
        elif pct < -0.1:
            directions[name] = "down"
        else:
            directions[name] = "flat"

    direction_counts: dict = {}
    for d in directions.values():
        direction_counts[d] = direction_counts.get(d, 0) + 1
    majority_count = max(direction_counts.values())
    direction_agreement = majority_count / len(directions)
    spread = max(pct_changes.values()) - min(pct_changes.values())

    # Heuristic thresholds (documented here, not derived from a formal
    # statistical test): "High" = every component agrees on direction and
    # their predicted % changes are all within 1 point of each other. "Low"
    # = fewer than 2/3 agree on direction, or the spread exceeds 3 points.
    if direction_agreement == 1.0 and spread <= 1.0:
        label = "High"
    elif direction_agreement < 0.67 or spread > 3.0:
        label = "Low"
    else:
        label = "Moderate"

    return {
        "direction_agreement": float(direction_agreement),
        "pct_change_spread": float(spread),
        "agreement_label": label,
        "component_pct_changes": {k: float(v) for k, v in pct_changes.items()},
        "component_directions": directions,
    }


def _representative_rmse(eval_dict: Optional[dict], wf_result: Optional[dict]) -> float:
    """
    Prefers walk-forward mean RMSE (a larger, more robust out-of-sample
    sample spanning several chronological folds) when it's actually usable;
    falls back to the single train/test split RMSE otherwise. Returns NaN
    if neither is usable, which callers must check for.
    """
    if wf_result is not None:
        n_folds = wf_result.get("n_folds", 0)
        mean_rmse = wf_result.get("mean_rmse", float("nan"))
        if n_folds >= 2 and np.isfinite(mean_rmse):
            return float(mean_rmse)
    if eval_dict is not None:
        rmse = eval_dict.get("rmse", float("nan"))
        if np.isfinite(rmse):
            return float(rmse)
    return float("nan")


def compute_prediction_confidence(ensemble_result: dict, lr_eval: dict, lstm_eval: dict,
                                   walk_forward_lr: dict = None,
                                   walk_forward_lstm: dict = None) -> dict:
    """
    Builds the confidence/uncertainty summary for a final ensemble
    prediction. See this module's docstring for what this is and isn't.

    Method:
      1. Model agreement (compute_model_agreement above).
      2. An approximate price range: final_predicted_price +/- a blend of
         each model's own representative RMSE (walk-forward mean where
         available, else the single split), weighted by the SAME
         linear_regression/lstm weights the ensemble itself used -- so a
         model that had more influence on the point prediction also has
         more influence on the width of the uncertainty band.
      3. A confidence label combining (1) and (2) via simple, documented
         heuristic rules.
    """
    component_predictions = ensemble_result["component_predictions"]
    current_price = ensemble_result["current_price"]
    final_price = ensemble_result["final_predicted_price"]

    agreement = compute_model_agreement(component_predictions, current_price)

    lr_rmse = _representative_rmse(lr_eval, walk_forward_lr)
    lstm_rmse = _representative_rmse(lstm_eval, walk_forward_lstm)

    weights = ensemble_result.get("weights_used") or {}
    w_lr = weights.get("linear_regression", 0.5)
    w_lstm = weights.get("lstm", 0.5)
    w_sum = w_lr + w_lstm

    have_lr = np.isfinite(lr_rmse)
    have_lstm = np.isfinite(lstm_rmse)
    if have_lr and have_lstm and w_sum > 0:
        blended_rmse = (w_lr * lr_rmse + w_lstm * lstm_rmse) / w_sum
    elif have_lr and have_lstm:
        blended_rmse = (lr_rmse + lstm_rmse) / 2
    elif have_lr:
        blended_rmse = lr_rmse
    elif have_lstm:
        blended_rmse = lstm_rmse
    else:
        blended_rmse = float("nan")

    if np.isfinite(blended_rmse):
        price_range = {"low": float(final_price - blended_rmse),
                        "high": float(final_price + blended_rmse)}
        band_pct = blended_rmse / current_price * 100 if current_price else None
    else:
        price_range = None
        band_pct = None

    predicted_move_pct = abs(ensemble_result["predicted_pct_change"])

    # Confidence label: documented heuristic, not a statistically calibrated
    # score. Low model agreement always caps confidence at "Low" -- if the
    # models don't even agree on direction, no error-band math should be
    # allowed to paper over that. Otherwise, an error band wider than the
    # predicted move itself (the model isn't even confident of the SIGN of
    # the move, in effect) also caps confidence at "Low".
    if agreement["agreement_label"] == "Low":
        confidence_label = "Low"
    elif band_pct is not None and predicted_move_pct > 0 and band_pct > predicted_move_pct * 1.5:
        confidence_label = "Low"
    elif agreement["agreement_label"] == "High" and (band_pct is None or band_pct < predicted_move_pct):
        confidence_label = "Moderate-High"
    else:
        confidence_label = "Moderate"

    return {
        "predicted_price": float(final_price),
        "expected_pct_change": float(ensemble_result["predicted_pct_change"]),
        "confidence_label": confidence_label,
        "model_agreement": agreement,
        "approximate_price_range": price_range,
        "approximate_error_basis": {
            "lr_representative_rmse": lr_rmse if have_lr else None,
            "lstm_representative_rmse": lstm_rmse if have_lstm else None,
            "blended_rmse": blended_rmse if np.isfinite(blended_rmse) else None,
            "source": ("walk-forward mean RMSE where >= 2 folds were available, "
                       "else the single train/test split RMSE"),
        },
        "methodology_note": (
            "APPROXIMATE uncertainty estimate, not a formally valid statistical "
            "confidence interval. The price range is the final prediction +/- a "
            "weighted blend of each model's own historical RMSE on held-out data "
            "(same weights the ensemble used for its point prediction). The "
            "confidence label is a documented heuristic combining that error "
            "magnitude with how much the component models agree with each other "
            "-- it is not derived from a calibrated probabilistic model or a "
            "backtested coverage rate, and should not be read as a statistical "
            "probability."
        ),
    }
