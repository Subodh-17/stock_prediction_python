"""
Tests for the live broker clients (Upstox, Groww) -- request payload
construction and response parsing, using mocked HTTP calls since this
sandbox has no network access to either broker and no real credentials.
"""
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trading.brokers.upstox_broker import UpstoxBroker
from trading.brokers.groww_broker import GrowwBroker

# ---------------------------------------------------------------------------
# UpstoxBroker
# ---------------------------------------------------------------------------
with patch("trading.brokers.upstox_broker.get_access_token", return_value="fake-upstox-token"):
    upstox = UpstoxBroker()
    assert upstox.access_token == "fake-upstox-token"
    print("Test 1 (UpstoxBroker: initializes with access token) passed")

    # -- funds --
    with patch("trading.brokers.upstox_broker.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": {"available_to_trade": {"total": 54321.0}}},
        )
        capital = upstox.get_available_capital()
        assert capital == 54321.0, capital
        called_headers = mock_get.call_args.kwargs["headers"]
        assert called_headers["Authorization"] == "Bearer fake-upstox-token"
        print("Test 2 (UpstoxBroker: get_available_capital parses funds response) passed:", capital)

    # -- positions --
    with patch("trading.brokers.upstox_broker.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"trading_symbol": "RELIANCE", "quantity": 12}]},
        )
        pos = upstox.get_position("RELIANCE.NS")
        assert pos == 12, pos
        print("Test 3 (UpstoxBroker: get_position matches normalized symbol) passed:", pos)

    # -- place order: success --
    with patch("trading.brokers.upstox_broker._resolve_instrument_key",
               return_value="NSE_EQ|INE002A01018"), \
         patch("trading.brokers.upstox_broker.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": {"order_ids": ["abc123"]}},
        )
        result = upstox.place_order("RELIANCE", "BUY", 5, price=2500)
        assert result["status"] == "OPEN", result
        assert result["broker_order_id"] == "abc123"
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["transaction_type"] == "BUY"
        assert sent_payload["quantity"] == 5
        assert sent_payload["instrument_token"] == "NSE_EQ|INE002A01018"
        print("Test 4 (UpstoxBroker: place_order builds correct payload, parses order_id) passed")

    # -- place order: instrument resolution failure -> ERROR, not a crash --
    with patch("trading.brokers.upstox_broker._resolve_instrument_key",
               side_effect=Exception("not found")):
        result = upstox.place_order("FAKESTOCK", "BUY", 5, price=100)
        assert result["status"] == "ERROR", result
        print("Test 5 (UpstoxBroker: unresolvable instrument -> ERROR, not exception) passed")

    # -- place order: unknown action rejected before any network call --
    with patch("trading.brokers.upstox_broker.requests.post") as mock_post:
        result = upstox.place_order("RELIANCE", "SHORT", 5, price=100)
        assert result["status"] == "REJECTED"
        mock_post.assert_not_called()
        print("Test 6 (UpstoxBroker: unknown action rejected without hitting the API) passed")

# ---------------------------------------------------------------------------
# GrowwBroker
# ---------------------------------------------------------------------------
with patch("trading.brokers.groww_broker.get_access_token", return_value="fake-groww-token"):
    groww = GrowwBroker()

    with patch("trading.brokers.groww_broker.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"payload": {"clear_cash": 78910.0}},
        )
        capital = groww.get_available_capital()
        assert capital == 78910.0, capital
        print("Test 7 (GrowwBroker: get_available_capital parses margin response) passed:", capital)

    with patch("trading.brokers.groww_broker.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"payload": {"holdings": [{"trading_symbol": "TCS", "quantity": 3}]}},
        )
        pos = groww.get_position("TCS")
        assert pos == 3, pos
        print("Test 8 (GrowwBroker: get_position parses holdings response) passed:", pos)

    with patch("trading.brokers.groww_broker.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "SUCCESS",
                          "payload": {"groww_order_id": "GRW123", "order_status": "open"}},
        )
        result = groww.place_order("TCS", "BUY", 2, price=3800)
        assert result["status"] == "OPEN", result
        assert result["broker_order_id"] == "GRW123"
        sent_payload = mock_post.call_args.kwargs["json"]
        assert sent_payload["transaction_type"] == "BUY"
        assert sent_payload["trading_symbol"] == "TCS"
        print("Test 9 (GrowwBroker: place_order builds correct payload, parses order_id) passed")

    with patch("trading.brokers.groww_broker.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "FAILURE", "error": "insufficient funds"},
        )
        result = groww.place_order("TCS", "BUY", 2, price=3800)
        assert result["status"] == "ERROR", result
        print("Test 10 (GrowwBroker: API-level failure surfaces as ERROR, not a crash) passed")

print("\nALL BROKER TESTS PASSED")
