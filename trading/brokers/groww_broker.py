"""
trading/brokers/groww_broker.py
-----------------------------------
Places REAL orders through your Groww trading account via the order/create
API. Like the Upstox broker, this is only ever reached after all of
trade_executor.py's safety gates have been explicitly satisfied.

Endpoints used:
  POST https://api.groww.in/v1/order/create           (place order)
  GET  https://api.groww.in/v1/margins/detail/user     (available cash)
  GET  https://api.groww.in/v1/holdings/user           (current holdings)

Every method here either returns cleanly or raises RuntimeError -- network
errors, timeouts, and malformed responses are all normalized to RuntimeError
rather than leaking raw `requests` exceptions, matching UpstoxBroker's
contract so trade_executor.py only ever needs to handle one exception type
from a broker regardless of which one is in use.
"""

import requests

import config
import security
from data_sources.groww_source import get_access_token
from .base_broker import BaseBroker

ORDER_URL = "https://api.groww.in/v1/order/create"
MARGIN_URL = "https://api.groww.in/v1/margins/detail/user"
HOLDINGS_URL = "https://api.groww.in/v1/holdings/user"
REQUEST_TIMEOUT = 30


class GrowwBroker(BaseBroker):
    broker_name = "groww"

    def __init__(self):
        try:
            self.access_token = get_access_token()
        except Exception as e:
            raise RuntimeError(f"Could not get a Groww access token: {security.safe_exception_text(e)}") from e

    def _headers(self):
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
            "X-API-VERSION": "1.0",
        }

    def get_available_capital(self) -> float:
        try:
            resp = requests.get(MARGIN_URL, headers=self._headers(), timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Groww margin request failed (network error): {security.safe_exception_text(e)}") from e
        if resp.status_code != 200:
            raise RuntimeError(f"Groww margin request failed: {security.safe_upstream_detail(resp.text, resp.status_code)}")
        try:
            payload = resp.json().get("payload", {})
            return float(payload.get("clear_cash", 0.0))
        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"Groww margin response was not in the expected shape: {security.safe_exception_text(e)}") from e

    def get_position(self, ticker: str) -> float:
        symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
        try:
            resp = requests.get(HOLDINGS_URL, headers=self._headers(), timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Groww holdings request failed (network error): {security.safe_exception_text(e)}") from e
        if resp.status_code != 200:
            raise RuntimeError(f"Groww holdings request failed: {security.safe_upstream_detail(resp.text, resp.status_code)}")
        try:
            holdings = resp.json().get("payload", {}).get("holdings", [])
            for h in holdings:
                if h.get("trading_symbol") == symbol:
                    return float(h.get("quantity", 0))
            return 0.0
        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"Groww holdings response was not in the expected shape: {security.safe_exception_text(e)}") from e

    def place_order(self, ticker: str, action: str, quantity: int, price: float = None,
                     product: str = "CNC", order_type: str = "MARKET", exchange: str = "NSE") -> dict:
        action = action.upper()
        if action not in ("BUY", "SELL"):
            return {"status": "REJECTED", "broker_order_id": "", "message": f"Unknown action '{action}'."}

        symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
        payload = {
            "validity": "DAY",
            "exchange": exchange,
            "transaction_type": action,
            "order_type": order_type,          # MARKET, LIMIT, SL, SL_M
            "price": 0 if order_type == "MARKET" else float(price or 0),
            "product": product,                 # CNC (delivery), MIS (intraday)
            "quantity": int(quantity),
            "segment": "CASH",
            "trading_symbol": symbol,
        }

        # This is a REAL order submission -- any network-level failure must
        # come back as a normal ERROR result, not an uncaught exception, so
        # trade_executor can still log what happened instead of crashing
        # with no audit trail of the attempt.
        try:
            resp = requests.post(
                ORDER_URL, json=payload,
                headers={**self._headers(), "Content-Type": "application/json"},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            return {"status": "ERROR", "broker_order_id": "",
                    "message": f"Groww order request failed (network error, order may or may not "
                               f""}

        if resp.status_code not in (200, 201):
            return {"status": "ERROR", "broker_order_id": "",
                    "message": f"Groww order request failed: {security.safe_upstream_detail(resp.text, resp.status_code)}"}

        try:
            body = resp.json()
        except ValueError as e:
            return {"status": "ERROR", "broker_order_id": "",
                    "message": f"Groww order was likely submitted but the response couldn't be "
                               f"parsed ({security.safe_exception_text(e)}) -- check your Groww account directly."}

        if body.get("status") != "SUCCESS":
            return {"status": "ERROR", "broker_order_id": "",
                    "message": f"Groww order error: {security.safe_upstream_detail(body.get('error'))}"}

        order_id = body.get("payload", {}).get("groww_order_id", "")
        order_status = body.get("payload", {}).get("order_status", "OPEN")
        return {"status": order_status.upper(), "broker_order_id": order_id,
                "message": f"[LIVE - GROWW] {action} order for {quantity} {ticker} submitted."}
