"""
ensemble.py
------------
Blends three independent sources of next-day price prediction into a single
final estimate:

  1. Linear Regression  -> a real dollar price prediction
  2. LSTM                -> a real dollar price prediction
  3. LLM panel            -> qualitative direction + estimated % change,
                             averaged across whichever providers responded

The quantitative models (LR, LSTM) get the bulk of the weight since they
actually see numerical price history. The LLM panel contributes a smaller
adjustment based on its reasoning over the trend summary -- useful as a
sanity check / qualitative second opinion, not a replacement for the
numerical models.

Weights are configurable in config.ENSEMBLE_WEIGHTS.
"""

import numpy as np
import config


def blend_llm_panel(llm_results: list, current_price: float) -> dict:
    """
    Combines individual LLM predictions into one panel estimate.
    Returns None if no LLM produced a valid result.
    """
    valid = [r for r in llm_results if r.get("error") is None]
    if not valid:
        return None

    # Confidence-weighted average of each LLM's estimated % change
    total_conf = sum(r["confidence"] for r in valid) or len(valid)
    weighted_pct = sum(r["estimated_pct_change"] * r["confidence"] for r in valid) / total_conf

    directions = [r["direction"] for r in valid]
    direction_votes = {d: directions.count(d) for d in set(directions)}
    majority_direction = max(direction_votes, key=direction_votes.get)

    panel_price = current_price * (1 + weighted_pct / 100)

    return {
        "panel_price_estimate": float(panel_price),
        "panel_pct_change": float(weighted_pct),
        "majority_direction": majority_direction,
        "num_providers": len(valid),
        "direction_votes": direction_votes,
        "individual_results": llm_results,
    }


def _split_lr_lstm_budget(lr_mean_rmse: float, lstm_mean_rmse: float,
                           llm_weight: float, min_weight_floor: float):
    """
    Pure inverse-RMSE weight-split math, factored out of
    compute_dynamic_weights() so backtest.py's historically-valid
    day-by-day re-weighting (see backtest.py's "ensemble" backtest) can
    reuse the EXACT same math against a different (per-day, prior-folds-only)
    RMSE pair, rather than a second, subtly-different implementation
    drifting out of sync with this one over time.

    Assumes lr_mean_rmse/lstm_mean_rmse are already known to be finite and
    positive (usability checks are the caller's responsibility -- this
    function only does the split itself).

    Returns (w_lr, w_lstm) such that w_lr + w_lstm == 1 - llm_weight exactly.
    """
    inv_lr = 1.0 / lr_mean_rmse
    inv_lstm = 1.0 / lstm_mean_rmse
    remaining_budget = 1.0 - llm_weight

    # Split remaining_budget (NOT the full 1.0) between LR/LSTM by inverse
    # RMSE, then apply the floor directly within that budget -- this keeps
    # w_lr + w_lstm exactly equal to remaining_budget at every step, so no
    # renormalization is needed afterwards. (An earlier version applied the
    # floor and then renormalized all three weights together, which could
    # push the floored weight back below the floor -- fixed here.)
    if 2 * min_weight_floor > remaining_budget:
        # The floor itself doesn't fit twice in the budget left after
        # reserving llm_weight -- an inherently infeasible configuration.
        # Fall back to splitting the remaining budget evenly rather than
        # producing a floor that can't actually be honored.
        return remaining_budget / 2, remaining_budget / 2

    raw_lr = remaining_budget * inv_lr / (inv_lr + inv_lstm)
    raw_lstm = remaining_budget - raw_lr
    if raw_lr < min_weight_floor:
        return min_weight_floor, remaining_budget - min_weight_floor
    if raw_lstm < min_weight_floor:
        return remaining_budget - min_weight_floor, min_weight_floor
    return raw_lr, raw_lstm


def compute_dynamic_weights(wf_lr, wf_lstm, llm_weight: float = None,
                             min_weight_floor: float = None,
                             min_folds_required: int = 3) -> dict:
    """
    Computes ensemble weights from walk-forward validation performance
    instead of the fixed config.ENSEMBLE_WEIGHTS.

    `wf_lr` / `wf_lstm` are validation.WalkForwardResult objects (or any
    object exposing .mean_rmse and .folds, a list).

    Method: inverse-RMSE weighting between LR and LSTM (the model with
    lower walk-forward RMSE -- i.e. that was more accurate on unseen,
    chronologically-later folds -- gets more weight), within whatever
    weight budget is left after reserving `llm_weight` for the LLM panel.
    The LLM panel's weight is NOT adjusted dynamically here (LLM
    predictions aren't backtested against a stored ground-truth history
    the way LR/LSTM are), which also satisfies the requirement that LLM
    output never gets to dominate the numerical models -- its influence is
    capped at the configured llm_weight regardless of how confident any
    individual LLM response sounds.

    Guardrails:
      - `min_weight_floor` (default config.ENSEMBLE_MIN_WEIGHT_FLOOR) is an
        absolute floor applied to both LR and LSTM after normalization, so
        a tiny/noisy validation sample can never push one model's weight to
        ~0% and the other to ~100%.
      - If either model doesn't have enough walk-forward folds
        (`min_folds_required`) or has a non-finite/non-positive mean RMSE,
        this function falls back to config.ENSEMBLE_WEIGHTS entirely
        (transparent, documented fallback -- never silently produces
        degenerate weights).

    Returns a dict with the same "linear_regression"/"lstm"/"llm_panel"
    keys as config.ENSEMBLE_WEIGHTS, plus "mode" and "basis" for
    transparency (what the weights were actually computed from).
    """
    llm_weight = llm_weight if llm_weight is not None else config.ENSEMBLE_WEIGHTS["llm_panel"]
    min_weight_floor = (min_weight_floor if min_weight_floor is not None
                         else config.ENSEMBLE_MIN_WEIGHT_FLOOR)

    def _usable(wf):
        return (wf is not None and len(getattr(wf, "folds", [])) >= min_folds_required
                and np.isfinite(getattr(wf, "mean_rmse", float("nan")))
                and wf.mean_rmse > 0)

    if not (_usable(wf_lr) and _usable(wf_lstm)) or llm_weight >= 1:
        fallback = dict(config.ENSEMBLE_WEIGHTS)
        fallback["mode"] = "fixed_fallback"
        fallback["basis"] = {
            "reason": (
                "Insufficient walk-forward validation folds (need >= "
                f"{min_folds_required} for both LR and LSTM) to compute "
                "reliable dynamic weights -- using configured fixed weights."
            )
        }
        return fallback

    w_lr, w_lstm = _split_lr_lstm_budget(wf_lr.mean_rmse, wf_lstm.mean_rmse,
                                          llm_weight, min_weight_floor)
    w_llm = llm_weight

    return {
        "linear_regression": float(w_lr),
        "lstm": float(w_lstm),
        "llm_panel": float(w_llm),
        "mode": "dynamic",
        "basis": {
            "lr_mean_rmse": wf_lr.mean_rmse,
            "lr_folds": len(wf_lr.folds),
            "lstm_mean_rmse": wf_lstm.mean_rmse,
            "lstm_folds": len(wf_lstm.folds),
            "min_weight_floor": min_weight_floor,
        },
    }


class AllModelsFailedError(Exception):
    """
    Raised when NOT ONE component model produced a usable prediction, so
    there is nothing left to blend. Deliberately an exception rather than a
    NaN result: a NaN would flow onward into signal_generator, the webapp,
    and report.json as if it were a real (just unusable) number, whereas
    this stops at the point the problem actually occurred and says so.
    """


def _is_usable(price) -> bool:
    """A component prediction is usable if it's a finite, positive number."""
    if price is None:
        return False
    try:
        value = float(price)
    except (TypeError, ValueError):
        return False
    return np.isfinite(value) and value > 0


def compute_ensemble_prediction(lr_price: float, lstm_price: float,
                                 llm_panel_summary: dict, current_price: float,
                                 weights: dict = None) -> dict:
    """
    Produces the final blended next-day price prediction.

    `weights` optionally overrides config.ENSEMBLE_WEIGHTS (e.g. with the
    output of compute_dynamic_weights()). Defaults to the fixed config
    weights, preserving prior behavior for existing callers.

    FAILED-COMPONENT HANDLING: a component whose prediction is missing
    (None) or not a finite positive price (NaN/inf/<=0 -- what a numerically
    blown-up model fit produces) is EXCLUDED from the blend, and its weight
    is redistributed proportionally across whichever components remain. It
    is not silently treated as 0, and it is not allowed to propagate into
    the blended figure: previously a single NaN from either model made the
    entire ensemble NaN, so one misbehaving model took down a prediction the
    other model could still have supported on its own.

    Exclusions are never hidden -- every excluded component is listed in the
    returned dict's "excluded_components", and its reported weight is 0.0.
    If NO component is usable, AllModelsFailedError is raised rather than
    returning a NaN price.
    """
    weights = dict(weights) if weights is not None else dict(config.ENSEMBLE_WEIGHTS)

    excluded = []
    if not _is_usable(lr_price):
        excluded.append({"component": "linear_regression", "value": _describe(lr_price)})
        lr_price = None
    if not _is_usable(lstm_price):
        excluded.append({"component": "lstm", "value": _describe(lstm_price)})
        lstm_price = None
    if llm_panel_summary is not None and not _is_usable(llm_panel_summary.get("panel_price_estimate")):
        excluded.append({"component": "llm_panel",
                          "value": _describe(llm_panel_summary.get("panel_price_estimate"))})
        llm_panel_summary = None

    if lr_price is None or lstm_price is None:
        return _blend_available_components(
            lr_price, lstm_price, llm_panel_summary, current_price, weights, excluded
        )

    if llm_panel_summary is None:
        # Redistribute the LLM weight proportionally across LR and LSTM
        remaining = weights["linear_regression"] + weights["lstm"]
        if remaining <= 0:
            # Only reachable with an unusual manual config (both LR_WEIGHT
            # and LSTM_WEIGHT set to 0 in .env) combined with no LLM panel
            # available (--skip-llm or no providers configured) -- fall back
            # to a plain average rather than dividing by zero.
            w_lr = w_lstm = 0.5
        else:
            w_lr = weights["linear_regression"] / remaining
            w_lstm = weights["lstm"] / remaining
        final_price = w_lr * lr_price + w_lstm * lstm_price
        components = {"linear_regression": lr_price, "lstm": lstm_price}
        # IMPORTANT: report the weights ACTUALLY applied above (post-
        # redistribution), not the original pre-redistribution `weights`
        # dict -- a real bug, found and fixed here: this used to always
        # return the original dict verbatim, so whenever no LLM panel was
        # available (--skip-llm, no API keys configured, or every provider
        # failing), "weights_used" would still claim e.g. 20% LLM / 35% LR /
        # 45% LSTM even though the LLM contributed literally 0% and its
        # budget had actually been redistributed to LR (43.75%) and LSTM
        # (56.25%). That misreported figure then propagated into every
        # consumer of weights_used: CLI/webapp display, report.json,
        # confidence.py's uncertainty-band weighting, and (new in this
        # pass) the ensemble backtest's weighting-mode="auto" detection.
        weights_used = dict(weights)  # keep any extra keys (mode/basis) for transparency
        weights_used["linear_regression"] = float(w_lr)
        weights_used["lstm"] = float(w_lstm)
        weights_used["llm_panel"] = 0.0
    else:
        w_lr = weights["linear_regression"]
        w_lstm = weights["lstm"]
        w_llm = weights["llm_panel"]
        final_price = (w_lr * lr_price + w_lstm * lstm_price +
                        w_llm * llm_panel_summary["panel_price_estimate"])
        components = {
            "linear_regression": lr_price,
            "lstm": lstm_price,
            "llm_panel": llm_panel_summary["panel_price_estimate"],
        }
        weights_used = weights

    pct_change_vs_current = (final_price - current_price) / current_price * 100

    return {
        "final_predicted_price": float(final_price),
        "current_price": float(current_price),
        "predicted_pct_change": float(pct_change_vs_current),
        "component_predictions": components,
        "weights_used": weights_used,
        "excluded_components": excluded,
    }


def _describe(value) -> str:
    """Short, safe description of an unusable component value, for reporting."""
    if value is None:
        return "missing"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return f"non-numeric ({type(value).__name__})"


def _blend_available_components(lr_price, lstm_price, llm_panel_summary,
                                 current_price: float, weights: dict,
                                 excluded: list) -> dict:
    """
    Blends whichever components ARE usable when at least one of LR/LSTM is
    not, redistributing the unusable components' weight proportionally
    across the survivors. Kept separate from the main two-model path above
    so that path's behavior is byte-for-byte what it always was whenever
    both quantitative models are healthy -- this function only ever runs in
    the degraded case.
    """
    available = {}
    if lr_price is not None:
        available["linear_regression"] = (float(lr_price), weights.get("linear_regression", 0.0))
    if lstm_price is not None:
        available["lstm"] = (float(lstm_price), weights.get("lstm", 0.0))
    if llm_panel_summary is not None:
        available["llm_panel"] = (float(llm_panel_summary["panel_price_estimate"]),
                                   weights.get("llm_panel", 0.0))

    if not available:
        raise AllModelsFailedError(
            "No component model produced a usable prediction "
            f"({', '.join(e['component'] for e in excluded)} all unusable), so there "
            "is nothing to blend. Refusing to emit a prediction rather than "
            "returning a meaningless number."
        )

    weight_total = sum(w for _, w in available.values())
    if weight_total <= 0:
        # Configured weights of the surviving components sum to zero -- fall
        # back to an equal split rather than dividing by zero.
        share = 1.0 / len(available)
        normalized = {name: share for name in available}
    else:
        normalized = {name: w / weight_total for name, (_, w) in available.items()}

    final_price = sum(available[name][0] * normalized[name] for name in available)

    weights_used = dict(weights)  # keep any extra keys (mode/basis) for transparency
    for name in ("linear_regression", "lstm", "llm_panel"):
        weights_used[name] = float(normalized.get(name, 0.0))

    pct_change_vs_current = (final_price - current_price) / current_price * 100
    return {
        "final_predicted_price": float(final_price),
        "current_price": float(current_price),
        "predicted_pct_change": float(pct_change_vs_current),
        "component_predictions": {name: available[name][0] for name in available},
        "weights_used": weights_used,
        "excluded_components": excluded,
    }
