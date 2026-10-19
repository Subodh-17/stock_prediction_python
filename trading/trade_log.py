"""
trading/trade_log.py
-----------------------
A simple append-only CSV audit log of every trade decision (including HOLDs)
and every order execution attempt (successful or not). This is what
risk_manager's daily trade-count limit reads from, and what you'd review to
answer "what did this thing actually do" after the fact.
"""

import csv
import os
from datetime import datetime, date, timedelta

import config

FIELDNAMES = [
    "timestamp", "ticker", "action", "mode", "broker",
    "predicted_pct_change", "current_price", "predicted_price",
    "quantity", "order_value", "status", "broker_order_id", "message",
]


def _ensure_log_exists(path: str = None):
    path = path or config.TRADE_LOG_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def _fmt_num(val, spec: str) -> str:
    if val is None:
        return ""
    try:
        f = float(val)
        return f"{f:{spec}}"
    except (ValueError, TypeError):
        return str(val)


def log_trade(ticker: str, action: str, mode: str, broker: str,
              predicted_pct_change: float, current_price: float, predicted_price: float,
              quantity: int = 0, order_value: float = 0.0, status: str = "N/A",
              broker_order_id: str = "", message: str = "", path: str = None):
    path = path or config.TRADE_LOG_PATH
    _ensure_log_exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker, "action": action, "mode": mode, "broker": broker,
            "predicted_pct_change": _fmt_num(predicted_pct_change, ".4f"),
            "current_price": _fmt_num(current_price, ".4f"), "predicted_price": _fmt_num(predicted_price, ".4f"),
            "quantity": quantity if quantity is not None else 0, "order_value": _fmt_num(order_value, ".2f"),
            "status": status, "broker_order_id": broker_order_id, "message": message,
        })


def count_trades_today(path: str = None) -> int:
    """
    Counts logged rows from today where an order was actually EXECUTED
    (status FILLED or OPEN) -- used to enforce the daily trade cap.

    Deliberately does NOT count BLOCKED/REJECTED/CANCELLED/ERROR attempts:
    the daily limit exists to cap real trading activity, not to penalize
    someone for a misconfigured order or a risk check doing its job. If
    blocked attempts counted too, a handful of failed attempts could lock
    out real trading for the rest of the day, which defeats the point of
    the limit.
    """
    path = path or config.TRADE_LOG_PATH
    if not os.path.exists(path):
        return 0
    today_str = date.today().isoformat()
    count = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["timestamp"].startswith(today_str)
                    and row["action"] in ("BUY", "SELL")
                    and row["status"] in ("FILLED", "OPEN")):
                count += 1
    return count


def read_all_trades(path: str = None) -> list:
    path = path or config.TRADE_LOG_PATH
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_recent_duplicate(ticker: str, action: str, quantity: int, mode: str,
                           window_seconds: int = None, path: str = None):
    """
    Returns the most recent logged order that is an exact duplicate of the
    one described -- same ticker, side, quantity and mode -- that was
    actually EXECUTED (status FILLED or OPEN) within `window_seconds`. Returns
    None when there is no such order.

    Reads the append-only audit log rather than keeping in-memory state, so
    the guard still works across separate CLI invocations and across a web
    server restart -- which is exactly when accidental double-submits happen.

    Rows with an unparseable timestamp are skipped rather than treated as
    matches: a corrupt line should not silently block legitimate trading.
    """
    window_seconds = (config.DUPLICATE_ORDER_WINDOW_SECONDS
                      if window_seconds is None else window_seconds)
    if window_seconds <= 0:
        return None

    path = path or config.TRADE_LOG_PATH
    if not os.path.exists(path):
        return None

    cutoff = datetime.now() - timedelta(seconds=window_seconds)
    newest = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("ticker") != ticker or row.get("action") != action.upper()
                    or row.get("mode") != mode or row.get("status") not in ("FILLED", "OPEN")):
                continue
            try:
                if int(float(row.get("quantity") or 0)) != int(quantity):
                    continue
                stamp = datetime.fromisoformat(row["timestamp"])
            except (ValueError, TypeError, KeyError):
                continue
            if stamp >= cutoff and (newest is None or stamp > newest[0]):
                newest = (stamp, row)
    return newest[1] if newest else None
