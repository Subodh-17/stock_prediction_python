"""
trading/brokers/paper_broker.py
-----------------------------------
A fully simulated broker -- no real money, no real orders, no network calls.
This is the DEFAULT broker for the whole trading module; you have to
explicitly opt into a real one (see trade_executor.py's safety gates).

State (cash + holdings) persists to a local JSON file between runs, so you
can track a simulated portfolio over multiple days/predictions the same way
you would a real one.
"""

import json
import os
import datetime as _dt

import config
from .base_broker import BaseBroker


class PaperBroker(BaseBroker):
    broker_name = "paper"

    def __init__(self, portfolio_path: str = None, starting_cash: float = None):
        self.portfolio_path = portfolio_path or config.PAPER_PORTFOLIO_PATH
        self.starting_cash = starting_cash or config.PAPER_STARTING_CASH
        self._load()

    def _load(self):
        if os.path.exists(self.portfolio_path):
            with open(self.portfolio_path) as f:
                self.state = json.load(f)
            # Backward-compatible: older portfolio files won't have this key.
            self.state.setdefault("avg_price", {})
        else:
            self.state = {"cash": self.starting_cash, "holdings": {}, "avg_price": {}}
            self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.portfolio_path) or ".", exist_ok=True)
        with open(self.portfolio_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def get_available_capital(self) -> float:
        return float(self.state["cash"])

    def get_position(self, ticker: str) -> float:
        return float(self.state["holdings"].get(ticker, 0))

    def get_avg_entry_price(self, ticker: str) -> float:
        """
        Average cost basis per share for the current open position in
        `ticker`, or 0.0 if there is no open position. Used by risk controls
        (stop-loss / take-profit) to know what price a position was entered
        at. Only tracked for the paper broker -- a live broker would need
        its own cost-basis lookup via that broker's position API (documented
        limitation; see README).
        """
        return float(self.state["avg_price"].get(ticker, 0.0))

    def get_day_start_value(self, current_total_value: float) -> float:
        """
        Returns today's starting mark-to-market portfolio value (cash +
        holdings), initializing it the first time this is called on a given
        calendar day (or if the stored value is from a previous day).
        `current_total_value` is only used to seed that value the first time
        it's needed today -- pass in cash + (shares * current price),
        computed by the caller (trade_executor.py).

        Used by risk_manager.check_daily_loss_limit() to gate new BUY
        orders once today's loss hits the configured DAILY_LOSS_LIMIT_PCT.
        """
        today_str = _dt.date.today().isoformat()
        day_start = self.state.get("day_start")
        if not day_start or day_start.get("date") != today_str:
            self.state["day_start"] = {"date": today_str, "value": current_total_value}
            self._save()
        return float(self.state["day_start"]["value"])

    def place_order(self, ticker: str, action: str, quantity: int, price: float = None) -> dict:
        action = action.upper()
        if quantity <= 0:
            return {"status": "REJECTED", "broker_order_id": "", "message": "Quantity must be positive."}
        if price is None or price <= 0:
            return {"status": "REJECTED", "broker_order_id": "", "message": "A valid price is required for the paper broker."}

        order_value = quantity * price

        if action == "BUY":
            if order_value > self.state["cash"] + 1e-6:
                return {"status": "REJECTED", "broker_order_id": "",
                        "message": f"Insufficient paper cash: need {order_value:.2f}, have {self.state['cash']:.2f}."}
            held_before = self.state["holdings"].get(ticker, 0)
            avg_before = self.state["avg_price"].get(ticker, 0.0)
            new_qty = held_before + quantity
            # Weighted-average cost basis across this buy and any existing position.
            self.state["avg_price"][ticker] = (
                (avg_before * held_before + price * quantity) / new_qty if new_qty else 0.0
            )
            self.state["cash"] -= order_value
            self.state["holdings"][ticker] = new_qty

        elif action == "SELL":
            held = self.state["holdings"].get(ticker, 0)
            if quantity > held + 1e-9:
                return {"status": "REJECTED", "broker_order_id": "",
                        "message": f"Insufficient paper holdings: trying to sell {quantity}, hold {held}."}
            self.state["cash"] += order_value
            rem = held - quantity
            if rem <= 1e-9:
                self.state["holdings"].pop(ticker, None)
                self.state["avg_price"].pop(ticker, None)
            else:
                self.state["holdings"][ticker] = rem

        else:
            return {"status": "REJECTED", "broker_order_id": "", "message": f"Unknown action '{action}'."}

        self._save()
        order_id = f"PAPER-{ticker}-{action}-{quantity}-{int(order_value)}"
        return {"status": "FILLED", "broker_order_id": order_id,
                "message": f"[PAPER] {action} {quantity} {ticker} @ {price:.2f} (simulated, no real money)."}
