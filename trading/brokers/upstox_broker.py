"""
trading/brokers/upstox_broker.py
------------------------------------
Places REAL orders through your Upstox trading account via the v3 Place
Order API. This is only ever reached when the multiple safety gates in
trade_executor.py have all been explicitly satisfied (ENABLE_LIVE_TRADING=true
in .env, --live on the CLI, and an interactive "yes I mean it" confirmation
per order) -- there is no path from a default run to a real order here.

Endpoints used:
  POST https://api-hft.upstox.com/v3/order/place   (place order)
  GET  https://api.upstox.com/v3/user/get-funds-and-margin  (available cash)
  GET  https://api.upstox.com/v2/portfolio/short-term-positions  (current holdings)

Every method here either returns cleanly or raises RuntimeError -- network
errors, timeouts, and malformed responses are all normalized to RuntimeError
rather than leaking raw `requests` exceptions, so trade_executor.py (and
anything else calling this class) only ever needs to handle one exception
type from a broker.
"""

import requests

import config
import security
from data_sources.upstox_source import get_access_token, _resolve_instrument_key
from .base_broker import BaseBroker

PLACE_ORDER_URL = "https://api-hft.upstox.com/v3/order/place"
FUNDS_URL = "https://api.upstox.com/v3/user/get-funds-and-margin"
POSITIONS_URL = "https://api.upstox.com/v2/portfolio/short-term-positions"
REQUEST_TIMEOUT = 30


class UpstoxBroker(BaseBroker):
    broker_name = "upstox"

    def __init__(self):
        try:
            self.access_token = get_access_token()
        except Exception as e:
            raise RuntimeError(f"Could not get an Upstox access token: {security.safe_exception_text(e)}") from e

    def _headers(self, extra: dict = None):
        headers = {"Authorization": f"Bearer {self.access_token}", "Accept": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    def get_available_capital(self) -> float:
        try:
            resp = requests.get(FUNDS_URL, headers=self._headers({"Api-Version": "3.0"}),
                                 timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Upstox funds request failed (network error): {security.safe_exception_text(e)}") from e
        if resp.status_code != 200:
            raise RuntimeError(f"Upstox funds request failed: {security.safe_upstream_detail(resp.text, resp.status_code)}")
        try:
            data = resp.json().get("data", {})
            return float(data.get("available_to_trade", {}).get("total", 0.0))
        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"Upstox funds response was not in the expected shape: {security.safe_exception_text(e)}") from e

    def get_position(self, ticker: str) -> float:
        symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
        try:
            resp = requests.get(POSITIONS_URL, headers=self._headers(), timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Upstox positions request failed (network error): {security.safe_exception_text(e)}") from e
        if resp.status_code != 200:
            raise RuntimeError(f"Upstox positions request failed: {security.safe_upstream_detail(resp.text, resp.status_code)}")
        try:
            positions = resp.json().get("data", [])
            for p in positions:
                if p.get("trading_symbol") == symbol:
                    return float(p.get("quantity", 0))
            return 0.0
        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"Upstox positions response was not in the expected shape: {security.safe_exception_text(e)}") from e

    def place_order(self, ticker: str, action: str, quantity: int, price: float = None,
                     product: str = "D", order_type: str = "MARKET", exchange: str = "NSE") -> dict:
        action = action.upper()
        if action not in ("BUY", "SELL"):
            return {"status": "REJECTED", "broker_order_id": "", "message": f"Unknown action '{action}'."}

        try:
            instrument_key = _resolve_instrument_key(ticker, exchange)
        except Exception as e:
            return {"status": "ERROR", "broker_order_id": "", "message": f"Could not resolve instrument: {security.safe_exception_text(e)}"}

        payload = {
            "quantity": int(quantity),
            "product": product,           # "D" = delivery, "I" = intraday
            "validity": "DAY",
            "price": 0 if order_type == "MARKET" else float(price or 0),
            "tag": "multi-ai-ensemble",
            "instrument_token": instrument_key,
            "order_type": order_type,      # MARKET, LIMIT, SL, SL-M
            "transaction_type": action,
            "disclosed_quantity": 0,
            "trigger_price": 0,
            "is_amo": False,
        }

        # This is a REAL order submission -- any network-level failure (timeout,
        # connection drop, etc.) must come back as a normal ERROR result, not an
        # uncaught exception, so trade_executor can still log what happened
        # instead of crashing with no audit trail of the attempt.
        try:
            resp = requests.post(
                PLACE_ORDER_URL, json=payload,
                headers=self._headers({"Content-Type": "application/json"}),
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            return {"status": "ERROR", "broker_order_id": "",
                    "message": f"Upstox order request failed (network error, order may or may not "
                               f""}

        if resp.status_code not in (200, 201):
            return {"status": "ERROR", "broker_order_id": "",
                    "message": f"Upstox order request failed: {security.safe_upstream_detail(resp.text, resp.status_code)}"}

        try:
            body = resp.json()
            order_ids = body.get("data", {}).get("order_ids", [])
            order_id = order_ids[0] if order_ids else ""
        except (ValueError, KeyError, TypeError) as e:
            return {"status": "ERROR", "broker_order_id": "",
                    "message": f"Upstox order was likely submitted but the response couldn't be "
                               f"parsed ({security.safe_exception_text(e)}) -- check your Upstox account directly."}

        return {"status": "OPEN", "broker_order_id": order_id,
                "message": f"[LIVE - UPSTOX] {action} order for {quantity} {ticker} submitted."}
