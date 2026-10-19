"""
trading/risk_manager.py
--------------------------
Applies hard, non-negotiable risk limits to every order before it's sent to
any broker (paper or live). These limits exist specifically so a model
misfire, a bad signal, or a bug can't wipe out an account -- they are
enforced in code, not just documented as a suggestion.
"""

import config
from .trade_log import count_trades_today


class RiskLimitError(Exception):
    """Raised when a proposed order would violate a configured risk limit."""
    pass


def compute_order_quantity(available_capital: float, current_price: float,
                            max_position_pct: float = None, max_order_value: float = None) -> int:
    """
    Computes how many whole shares/units to BUY given the capital allocated
    to this trade, respecting both the max-% -of-capital rule and the
    absolute max-order-value cap (whichever is more restrictive wins).
    Returns 0 if the price is too high to afford even 1 share within limits.

    This is a BUY-only sizing function: it answers "how much can I afford",
    which is meaningless for a SELL (you're not spending capital, you're
    converting shares you already hold into cash). Use
    compute_sell_quantity() for SELL orders.
    """
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if available_capital <= 0:
        return 0

    max_position_pct = config.MAX_POSITION_PCT if max_position_pct is None else max_position_pct
    max_order_value = config.MAX_ORDER_VALUE if max_order_value is None else max_order_value

    budget = min(available_capital * max_position_pct, max_order_value, available_capital)
    quantity = int(budget // current_price)  # floor -- no fractional shares
    return max(quantity, 0)


def compute_sell_quantity(held_quantity: float, current_price: float,
                           requested_quantity: int = None, max_order_value: float = None) -> int:
    """
    Computes how many whole shares/units to SELL. Unlike a BUY, this is sized
    against what you actually hold, not against available cash -- you can
    never sell more than held_quantity, full stop, regardless of any other
    limit. max_order_value still applies as a "don't dump a huge position in
    one order" cap, same spirit as the BUY-side cap.

    Defaults to selling the entire held position if no requested_quantity is
    given (a SELL signal means "exit the position", not "sell one share").
    """
    held_quantity = int(held_quantity)
    if held_quantity <= 0:
        return 0

    max_order_value = config.MAX_ORDER_VALUE if max_order_value is None else max_order_value
    max_qty_by_value = int(max_order_value // current_price) if current_price > 0 else held_quantity

    quantity = held_quantity if requested_quantity is None else min(requested_quantity, held_quantity)
    # Always apply the value cap, including when it caps down to 0 (e.g. a
    # single share already costs more than MAX_ORDER_VALUE) -- previously
    # this was skipped whenever max_qty_by_value was 0, which let an
    # oversized quantity slip through compute_sell_quantity uncapped, only
    # to be rejected later by validate_order instead of being cleanly sized
    # to 0 here.
    quantity = min(quantity, max_qty_by_value)
    return max(int(quantity), 0)


def check_daily_trade_limit(max_trades_per_day: int = None):
    """Raises RiskLimitError if today's trade count already hit the cap."""
    max_trades_per_day = config.MAX_TRADES_PER_DAY if max_trades_per_day is None else max_trades_per_day
    trades_today = count_trades_today()
    if trades_today >= max_trades_per_day:
        raise RiskLimitError(
            f"Daily trade limit reached ({trades_today}/{max_trades_per_day} trades "
            f"already logged today). Raise MAX_TRADES_PER_DAY in .env if this is intentional."
        )


def check_stop_loss_take_profit(entry_price: float, current_price: float,
                                 stop_loss_pct: float = None, take_profit_pct: float = None):
    """
    Returns "STOP_LOSS", "TAKE_PROFIT", or None.

    stop_loss_pct / take_profit_pct are POSITIVE percentages (e.g. 5 means
    "exit if the position is down 5% from entry" / "exit if up 10%"). Either
    defaults to config.STOP_LOSS_PCT / config.TAKE_PROFIT_PCT, both of which
    default to 0 (disabled) -- so calling this with no arguments is a no-op
    unless the user has opted in via .env.

    If both would somehow trigger simultaneously (only possible with
    contradictory config, e.g. stop_loss_pct describing a smaller move than
    take_profit_pct in the wrong direction), TAKE_PROFIT is checked first
    since it's the less risky of the two outcomes to act on by mistake.
    """
    stop_loss_pct = config.STOP_LOSS_PCT if stop_loss_pct is None else stop_loss_pct
    take_profit_pct = config.TAKE_PROFIT_PCT if take_profit_pct is None else take_profit_pct
    if entry_price <= 0:
        return None
    pct_change = (current_price - entry_price) / entry_price * 100
    if take_profit_pct > 0 and pct_change >= take_profit_pct:
        return "TAKE_PROFIT"
    if stop_loss_pct > 0 and pct_change <= -stop_loss_pct:
        return "STOP_LOSS"
    return None


def check_portfolio_exposure(proposed_order_value: float, existing_position_value: float,
                              total_portfolio_value: float, max_exposure_pct: float = None):
    """
    Raises RiskLimitError if (existing_position_value + proposed_order_value)
    would exceed max_exposure_pct of total_portfolio_value.

    Distinct from MAX_POSITION_PCT (which caps a single order against
    available capital): this caps the TOTAL value that ends up committed to
    a position, existing + new, against the whole portfolio's value.
    Defaults to config.MAX_PORTFOLIO_EXPOSURE_PCT, which defaults to 0
    (disabled) -- a no-op unless the user opts in via .env.
    """
    max_exposure_pct = config.MAX_PORTFOLIO_EXPOSURE_PCT if max_exposure_pct is None else max_exposure_pct
    if max_exposure_pct <= 0 or total_portfolio_value <= 0:
        return
    projected = existing_position_value + proposed_order_value
    cap = total_portfolio_value * (max_exposure_pct / 100)
    if projected > cap + 1e-6:
        raise RiskLimitError(
            f"Proposed position value ({projected:.2f}) would exceed "
            f"MAX_PORTFOLIO_EXPOSURE_PCT ({max_exposure_pct:.1f}% = {cap:.2f} of a "
            f"{total_portfolio_value:.2f} portfolio)."
        )


def check_daily_loss_limit(portfolio_value_now: float, portfolio_value_day_start: float,
                            daily_loss_limit_pct: float = None):
    """
    Raises RiskLimitError if the portfolio's mark-to-market value has
    already dropped by daily_loss_limit_pct or more since the start of
    today. Defaults to config.DAILY_LOSS_LIMIT_PCT, which defaults to 0
    (disabled).

    Only intended to gate NEW BUY orders -- callers should not use this to
    block SELLs (reducing risk should never be blocked by a loss limit).
    """
    daily_loss_limit_pct = config.DAILY_LOSS_LIMIT_PCT if daily_loss_limit_pct is None else daily_loss_limit_pct
    if daily_loss_limit_pct <= 0 or portfolio_value_day_start <= 0:
        return
    pct_change = (portfolio_value_now - portfolio_value_day_start) / portfolio_value_day_start * 100
    if pct_change <= -daily_loss_limit_pct:
        raise RiskLimitError(
            f"Daily loss limit reached: portfolio is down {pct_change:.2f}% today "
            f"(vs. the configured DAILY_LOSS_LIMIT_PCT of {daily_loss_limit_pct:.1f}%). "
            f"New BUY orders are blocked for the rest of today; SELLs are unaffected."
        )


def validate_order(quantity: int, current_price: float, available_capital: float,
                    action: str = "BUY", held_quantity: float = 0,
                    max_position_pct: float = None, max_order_value: float = None):
    """
    Final sanity check right before submission: re-derives the max allowed
    quantity from the same limits and raises RiskLimitError if the proposed
    order exceeds it. This is a second, independent check even if the
    quantity came from compute_order_quantity()/compute_sell_quantity() --
    defense in depth against a caller passing in a hand-picked quantity that
    skips the sizing logic.

    BUY and SELL are validated against different limits (see
    compute_order_quantity vs compute_sell_quantity for why): a BUY can never
    exceed available capital; a SELL can never exceed held_quantity. Both are
    still capped by max_order_value.
    """
    action = action.upper()
    if quantity <= 0:
        raise RiskLimitError("Order quantity must be a positive integer.")

    order_value = quantity * current_price
    max_order_value = config.MAX_ORDER_VALUE if max_order_value is None else max_order_value

    if action == "SELL":
        if quantity > held_quantity + 1e-6:
            raise RiskLimitError(
                f"Cannot sell {quantity} shares -- only {held_quantity} held."
            )
        if order_value > max_order_value + 1e-6:
            raise RiskLimitError(
                f"Order value ({order_value:.2f}) exceeds MAX_ORDER_VALUE ({max_order_value:.2f})."
            )
        return

    # BUY
    max_position_pct = config.MAX_POSITION_PCT if max_position_pct is None else max_position_pct
    cap = min(available_capital * max_position_pct, max_order_value)
    if order_value > cap + 1e-6:  # small epsilon for float rounding
        raise RiskLimitError(
            f"Order value ({order_value:.2f}) exceeds the allowed cap ({cap:.2f}) -- "
            f"{max_position_pct:.0%} of capital or {max_order_value:.2f} absolute, whichever is lower."
        )
    if order_value > available_capital + 1e-6:
        raise RiskLimitError(
            f"Order value ({order_value:.2f}) exceeds available capital ({available_capital:.2f})."
        )
