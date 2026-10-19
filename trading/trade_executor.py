"""
trading/trade_executor.py
----------------------------
Orchestrates: signal -> risk checks -> (for live mode) explicit confirmation
-> broker.place_order() -> audit log. This is the ONLY place in the project
that can cause a real order to be placed, and it is deliberately designed
so that happens only when every one of these is true:

  1. mode == "live" was explicitly requested (default is always "paper")
  2. config.ENABLE_LIVE_TRADING is True (must be set by hand in .env --
     there is no CLI flag that sets this; it has to be a deliberate,
     separate edit to a config file, not something a single flag typo
     could trigger)
  3. The daily trade limit (risk_manager) hasn't been hit
  4. The order passes risk_manager's position-sizing validation
  5. EITHER the caller passes auto_confirm=True explicitly (meant for
     pre-reviewed automated/scheduled runs) OR the person interactively
     types the exact confirmation phrase when prompted

Every decision -- including HOLDs, and including rejected/blocked orders --
is written to the audit log via trade_log.py.
"""

import config
from . import signal_generator, risk_manager, trade_log
from .brokers.paper_broker import PaperBroker

DISCLAIMER = (
    "\n" + "!" * 70 +
    "\nThis is an educational tool, NOT financial advice. Predictions come\n"
    "from statistical models (Linear Regression, LSTM) and LLM opinions --\n"
    "none of them can reliably predict markets. Past performance does not\n"
    "guarantee future results. Trading -- especially live, real-money\n"
    "trading -- carries real risk of loss. You are solely responsible for\n"
    "any trades executed through this tool.\n" + "!" * 70 + "\n"
)

CONFIRMATION_PHRASE = "CONFIRM LIVE TRADE"


def _get_broker(mode: str, broker_name: str):
    if mode == "paper":
        return PaperBroker()

    if not config.ENABLE_LIVE_TRADING:
        raise PermissionError(
            "Live trading is disabled. Set ENABLE_LIVE_TRADING=true in your "
            ".env file to enable it -- this is a deliberate, separate step "
            "from any CLI flag, by design."
        )

    if broker_name == "upstox":
        from .brokers.upstox_broker import UpstoxBroker
        return UpstoxBroker()
    elif broker_name == "groww":
        from .brokers.groww_broker import GrowwBroker
        return GrowwBroker()
    else:
        raise ValueError(f"Unknown live broker '{broker_name}'. Use 'upstox' or 'groww'.")


def execute_trade_decision(ticker: str, final_ensemble: dict, mode: str = "paper",
                            broker_name: str = None, capital_override: float = None,
                            quantity_override: int = None, auto_confirm: bool = False,
                            verbose: bool = True) -> dict:
    """
    Runs the full decision -> execution -> logging flow for one ticker.
    Returns a dict describing what happened (always succeeds in the sense
    of returning a result, even if the outcome was "blocked" or "rejected"
    -- exceptions are reserved for programming errors, not normal trading
    outcomes like insufficient funds).
    """
    mode = mode.lower()
    broker_name = broker_name or config.DEFAULT_BROKER

    # Guard against a misleading audit log: paper mode ALWAYS uses the paper
    # broker, regardless of what broker_name was passed in. Without this,
    # e.g. mode="paper", broker_name="upstox" would silently execute against
    # the paper broker while logging "broker: upstox" -- correct behavior,
    # wrong record of what actually happened.
    if mode == "paper" and broker_name != "paper":
        if verbose:
            print(f"[trade_executor] Note: mode=paper always uses the paper broker "
                  f"(ignoring broker_name='{broker_name}').")
        broker_name = "paper"

    if verbose:
        print(DISCLAIMER)

    try:
        signal = signal_generator.generate_signal(final_ensemble)
    except ValueError as e:
        # Covers both a NaN prediction and misconfigured (inverted)
        # thresholds -- either way, this is a real problem worth logging,
        # not something that should crash the run with no record of it.
        current_price = final_ensemble.get("current_price", 0.0)
        trade_log.log_trade(
            ticker=ticker, action="ERROR", mode=mode, broker=broker_name,
            predicted_pct_change=final_ensemble.get("predicted_pct_change", float("nan")),
            current_price=current_price, predicted_price=final_ensemble.get("final_predicted_price", 0.0),
            status="ERROR", message=str(e),
        )
        if verbose:
            print(f"ERROR generating signal: {e}")
        return {"executed": False, "reason": str(e), "signal": None}

    if verbose:
        print(f"Signal for {ticker}: {signal['action']} -- {signal['reasoning']}")

    # --- Stop-loss / take-profit override ---
    # Checked against any existing position BEFORE the HOLD short-circuit,
    # so a triggered stop-loss/take-profit can force a SELL even if the
    # model's own signal was HOLD (or even BUY). Only supported for the
    # paper broker today, since it's the only broker this project tracks an
    # average entry price for locally (see PaperBroker.get_avg_entry_price).
    # A no-op (both STOP_LOSS_PCT and TAKE_PROFIT_PCT default to 0/disabled)
    # unless explicitly configured.
    if mode == "paper":
        _pb_check = PaperBroker()
        _held_for_check = _pb_check.get_position(ticker)
        if _held_for_check > 0:
            _entry_price = _pb_check.get_avg_entry_price(ticker)
            _trigger = risk_manager.check_stop_loss_take_profit(_entry_price, signal["current_price"])
            if _trigger:
                signal["action"] = "SELL"
                signal["reasoning"] = (
                    f"{_trigger} triggered (entry {_entry_price:.2f} -> current "
                    f"{signal['current_price']:.2f}) -- overriding the model's signal to SELL."
                )
                if verbose:
                    print(f"[risk] {signal['reasoning']}")

    if signal["action"] == "HOLD":
        trade_log.log_trade(
            ticker=ticker, action="HOLD", mode=mode, broker=broker_name,
            predicted_pct_change=signal["predicted_pct_change"],
            current_price=signal["current_price"], predicted_price=signal["predicted_price"],
            status="N/A", message=signal["reasoning"],
        )
        return {"executed": False, "reason": "HOLD signal -- no order placed.", "signal": signal}

    try:
        broker = _get_broker(mode, broker_name)
    except (PermissionError, ValueError, RuntimeError) as e:
        trade_log.log_trade(
            ticker=ticker, action=signal["action"], mode=mode, broker=broker_name,
            predicted_pct_change=signal["predicted_pct_change"],
            current_price=signal["current_price"], predicted_price=signal["predicted_price"],
            status="BLOCKED", message=str(e),
        )
        if verbose:
            print(f"BLOCKED: {e}")
        return {"executed": False, "reason": str(e), "signal": signal}

    # --- risk checks (daily limit + position sizing) ---
    try:
        risk_manager.check_daily_trade_limit()
        available_capital = capital_override if capital_override is not None else broker.get_available_capital()
        held_quantity = broker.get_position(ticker)

        if signal["action"] == "BUY":
            # Daily loss limit: blocks new BUYs (never SELLs) once today's
            # mark-to-market loss hits DAILY_LOSS_LIMIT_PCT. Paper-broker
            # only (only broker this project tracks a day-start value for).
            # No-op when DAILY_LOSS_LIMIT_PCT is 0 (the default).
            if mode == "paper" and config.DAILY_LOSS_LIMIT_PCT > 0:
                current_total_value = available_capital + held_quantity * signal["current_price"]
                day_start_value = broker.get_day_start_value(current_total_value)
                risk_manager.check_daily_loss_limit(current_total_value, day_start_value)

            if quantity_override is not None:
                quantity = quantity_override
            else:
                quantity = risk_manager.compute_order_quantity(available_capital, signal["current_price"])

            # Portfolio exposure cap: existing position value + this
            # proposed order must not exceed MAX_PORTFOLIO_EXPOSURE_PCT of
            # total portfolio value. No-op when it's 0 (the default).
            if config.MAX_PORTFOLIO_EXPOSURE_PCT > 0 and quantity > 0:
                existing_position_value = held_quantity * signal["current_price"]
                total_portfolio_value = available_capital + existing_position_value
                risk_manager.check_portfolio_exposure(
                    proposed_order_value=quantity * signal["current_price"],
                    existing_position_value=existing_position_value,
                    total_portfolio_value=total_portfolio_value,
                )
        else:  # SELL -- sized against what's actually held, never against capital
            quantity = risk_manager.compute_sell_quantity(
                held_quantity, signal["current_price"], requested_quantity=quantity_override,
            )

        if quantity <= 0:
            if signal["action"] == "SELL":
                reason = (f"No position to sell (0 shares of {ticker} held)." if held_quantity <= 0
                           else f"Computed sell quantity is 0.")
            else:
                reason = (f"Computed order quantity is 0 (available capital {available_capital:.2f} "
                           f"too low relative to price {signal['current_price']:.2f} and risk limits).")
            raise risk_manager.RiskLimitError(reason)

        risk_manager.validate_order(quantity, signal["current_price"], available_capital,
                                     action=signal["action"], held_quantity=held_quantity)

    except (risk_manager.RiskLimitError, RuntimeError) as e:
        trade_log.log_trade(
            ticker=ticker, action=signal["action"], mode=mode, broker=broker_name,
            predicted_pct_change=signal["predicted_pct_change"],
            current_price=signal["current_price"], predicted_price=signal["predicted_price"],
            status="BLOCKED", message=str(e),
        )
        if verbose:
            print(f"BLOCKED: {e}")
        return {"executed": False, "reason": str(e), "signal": signal}

    # --- duplicate-order protection (LIVE only) ---
    # Checked after sizing (so the comparison is against the exact quantity
    # that would be submitted) and before the confirmation prompt, so a
    # double-submit is refused rather than asking the user to confirm an
    # order that should not be placed at all.
    if mode == "live":
        recent = trade_log.find_recent_duplicate(
            ticker=ticker, action=signal["action"], quantity=quantity, mode=mode,
        )
        if recent is not None:
            reason = (
                f"Duplicate order blocked: an identical live {signal['action']} of "
                f"{quantity} {ticker} was already executed at {recent['timestamp']} "
                f"(within the {config.DUPLICATE_ORDER_WINDOW_SECONDS}s duplicate window). "
                f"Set DUPLICATE_ORDER_WINDOW_SECONDS=0 in .env if you genuinely intend "
                f"to place repeated identical orders."
            )
            trade_log.log_trade(
                ticker=ticker, action=signal["action"], mode=mode, broker=broker_name,
                predicted_pct_change=signal["predicted_pct_change"],
                current_price=signal["current_price"], predicted_price=signal["predicted_price"],
                quantity=quantity, order_value=quantity * signal["current_price"],
                status="BLOCKED", message=reason,
            )
            if verbose:
                print(f"BLOCKED: {reason}")
            return {"executed": False, "reason": reason, "signal": signal}

    order_value = quantity * signal["current_price"]
    if verbose:
        print(f"Proposed order: {signal['action']} {quantity} x {ticker} "
              f"@ ~{signal['current_price']:.2f} (value ~{order_value:.2f}) via {broker_name} ({mode})")

    # --- confirmation gate for live orders ---
    if mode == "live" and not auto_confirm:
        typed = input(f"Type '{CONFIRMATION_PHRASE}' to submit this REAL order, anything else to cancel: ").strip()
        if typed != CONFIRMATION_PHRASE:
            trade_log.log_trade(
                ticker=ticker, action=signal["action"], mode=mode, broker=broker_name,
                predicted_pct_change=signal["predicted_pct_change"],
                current_price=signal["current_price"], predicted_price=signal["predicted_price"],
                quantity=quantity, order_value=order_value,
                status="CANCELLED", message="User did not confirm.",
            )
            if verbose:
                print("Order cancelled -- confirmation phrase did not match.")
            return {"executed": False, "reason": "Not confirmed by user.", "signal": signal}

    # --- execute ---
    # This call is wrapped even though the broker classes already normalize
    # their own network errors into clean {"status": "ERROR", ...} dicts --
    # this is the last line of defense so that ANY unexpected exception here
    # (a bug, a broker implementation that doesn't follow the contract, etc.)
    # still results in a logged audit entry instead of silently vanishing
    # with no record that an order was ever attempted.
    try:
        result = broker.place_order(ticker, signal["action"], quantity, price=signal["current_price"])
    except Exception as e:
        trade_log.log_trade(
            ticker=ticker, action=signal["action"], mode=mode, broker=broker_name,
            predicted_pct_change=signal["predicted_pct_change"],
            current_price=signal["current_price"], predicted_price=signal["predicted_price"],
            quantity=quantity, order_value=order_value,
            status="ERROR", message=f"Unexpected error placing order: {e}",
        )
        if verbose:
            print(f"ERROR placing order: {e}")
        return {"executed": False, "reason": f"Unexpected error placing order: {e}", "signal": signal}

    trade_log.log_trade(
        ticker=ticker, action=signal["action"], mode=mode, broker=broker_name,
        predicted_pct_change=signal["predicted_pct_change"],
        current_price=signal["current_price"], predicted_price=signal["predicted_price"],
        quantity=quantity, order_value=order_value,
        status=result["status"], broker_order_id=result["broker_order_id"], message=result["message"],
    )
    if verbose:
        print(result["message"])

    return {
        "executed": result["status"] in ("FILLED", "OPEN"),
        "reason": result["message"],
        "signal": signal,
        "quantity": quantity,
        "order_value": order_value,
        "broker_result": result,
    }
