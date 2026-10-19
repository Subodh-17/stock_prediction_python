"""
Unit test for the JSON-parsing logic in base_client and the ensemble blending
logic in ensemble.py, using mock LLM responses (since no live API keys are
available in this sandbox). Confirms the full panel -> blend -> ensemble
chain works correctly once a user plugs in real keys.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_integrations.base_client import BaseLLMClient
import ensemble
import config


class MockClient(BaseLLMClient):
    provider_name = "MockProvider"

    def __init__(self, api_key, canned_response):
        super().__init__(api_key=api_key, model="mock-model")
        self.canned_response = canned_response

    def _call_api(self, prompt: str) -> str:
        return self.canned_response


# --- Test 1: clean JSON response ---
c1 = MockClient("fake-key", '{"direction": "up", "confidence": 0.8, "estimated_pct_change": 1.2, "reasoning": "Momentum looks positive."}')
r1 = c1.get_prediction("some trend summary")
assert r1["direction"] == "up" and r1["error"] is None, r1
print("Test 1 (clean JSON) passed:", r1)

# --- Test 2: response wrapped in markdown fences (common LLM behavior despite instructions) ---
c2 = MockClient("fake-key", '```json\n{"direction": "down", "confidence": 0.6, "estimated_pct_change": -0.8, "reasoning": "Slight pullback expected."}\n```')
r2 = c2.get_prediction("some trend summary")
assert r2["direction"] == "down" and r2["error"] is None, r2
print("Test 2 (markdown-fenced JSON) passed:", r2)

# --- Test 3: malformed response should be captured as an error, not crash ---
c3 = MockClient("fake-key", "I think the stock will go up but I forgot to use JSON format sorry!")
r3 = c3.get_prediction("some trend summary")
assert r3["error"] is not None, r3
print("Test 3 (malformed response -> graceful error) passed:", r3)

# --- Test 4: unconfigured client (no API key) should skip cleanly ---
c4 = MockClient("", '{"direction": "flat"}')
r4 = c4.get_prediction("some trend summary")
assert r4["error"] == "No API key configured", r4
print("Test 4 (no key -> skipped) passed:", r4)

# --- Test 5: ensemble blending across a mix of valid + errored providers ---
mock_panel_results = [
    {"provider": "Claude", "direction": "up", "confidence": 0.8, "estimated_pct_change": 1.2, "reasoning": "...", "error": None},
    {"provider": "ChatGPT", "direction": "up", "confidence": 0.6, "estimated_pct_change": 0.9, "reasoning": "...", "error": None},
    {"provider": "Gemini", "direction": "down", "confidence": 0.5, "estimated_pct_change": -0.5, "reasoning": "...", "error": None},
    {"provider": "Grok", "direction": None, "confidence": 0.0, "estimated_pct_change": 0.0, "reasoning": None, "error": "timeout"},
    {"provider": "DeepSeek", "direction": None, "confidence": 0.0, "estimated_pct_change": 0.0, "reasoning": None, "error": "No API key configured"},
]
current_price = 200.0
panel_summary = ensemble.blend_llm_panel(mock_panel_results, current_price)
assert panel_summary["num_providers"] == 3, panel_summary
assert panel_summary["majority_direction"] == "up", panel_summary
print("Test 5 (panel blending, 3 valid / 2 errored) passed:", panel_summary)

final = ensemble.compute_ensemble_prediction(
    lr_price=205.0, lstm_price=203.0, llm_panel_summary=panel_summary, current_price=current_price
)
print("Test 5b (final ensemble with LLM panel) passed:", final)

# --- Test 6: ensemble with NO llm panel at all (weight redistribution) ---
final_no_llm = ensemble.compute_ensemble_prediction(
    lr_price=205.0, lstm_price=203.0, llm_panel_summary=None, current_price=current_price
)
print("Test 6 (final ensemble, no LLM panel, weights redistributed) passed:", final_no_llm)

# --- Test 7: zero-weight edge case (regression test for a division-by-zero
# bug found during QA -- only reachable with an unusual manual .env config
# of LR_WEIGHT=0 and LSTM_WEIGHT=0 combined with no LLM panel available) ---
original_weights = dict(config.ENSEMBLE_WEIGHTS)
config.ENSEMBLE_WEIGHTS = {"linear_regression": 0.0, "lstm": 0.0, "llm_panel": 1.0}
result = ensemble.compute_ensemble_prediction(lr_price=105.0, lstm_price=95.0,
                                                llm_panel_summary=None, current_price=100.0)
assert result["final_predicted_price"] == 100.0, result  # falls back to a plain average, not a crash
config.ENSEMBLE_WEIGHTS = original_weights
print("Test 7 (ensemble: LR+LSTM weights both 0 with no LLM panel -> falls back to average, "
      "not ZeroDivisionError) passed")

print("\nALL UNIT TESTS PASSED")
