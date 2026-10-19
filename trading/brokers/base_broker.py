"""
trading/brokers/base_broker.py
----------------------------------
Common interface every broker (paper, Upstox, Groww) implements, so
trade_executor.py never needs to know which one it's talking to.
"""

from abc import ABC, abstractmethod


class BaseBroker(ABC):
    broker_name = "base"

    @abstractmethod
    def get_available_capital(self) -> float:
        """Returns cash available to trade with, in the account's currency."""
        raise NotImplementedError

    @abstractmethod
    def get_position(self, ticker: str) -> float:
        """Returns current held quantity of `ticker` (0 if none)."""
        raise NotImplementedError

    @abstractmethod
    def place_order(self, ticker: str, action: str, quantity: int, price: float = None) -> dict:
        """
        Places a BUY or SELL order. Returns a normalized dict:
        {status, broker_order_id, message}
        status is one of: "FILLED", "OPEN", "REJECTED", "ERROR"
        """
        raise NotImplementedError
