"""
ai_integrations/panel.py
--------------------------
Queries all five LLM providers concurrently and collects their normalized
predictions. Providers without an API key configured are skipped (not
treated as failures) so the project runs fine with only 1, 2, or all 5 keys
supplied.
"""

import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import config
from .claude_client import ClaudeClient
from .openai_client import OpenAIClient
from .gemini_client import GeminiClient
from .grok_client import GrokClient
from .deepseek_client import DeepSeekClient


def get_all_clients():
    return [ClaudeClient(), OpenAIClient(), GeminiClient(), GrokClient(), DeepSeekClient()]


def query_llm_panel(trend_summary: str, verbose: bool = True) -> list:
    """
    Returns a list of prediction dicts, one per provider that is configured.
    Providers with no API key are reported but excluded from the ensemble math.

    Each provider already sets its own ~30s HTTP timeout (see the individual
    *_client.py files); the future.result(timeout=...) below is a second,
    independent ceiling at the panel level so that even an SDK/network edge
    case that doesn't honor its own timeout can't hang the whole pipeline --
    it just gets reported as a timed-out provider, same as any other error.
    """
    clients = get_all_clients()
    configured = [c for c in clients if c.is_configured()]
    skipped = [c.provider_name for c in clients if not c.is_configured()]

    if verbose and skipped:
        print(f"[panel] Skipping providers with no API key set: {', '.join(skipped)}")

    if not configured:
        if verbose:
            print("[panel] No LLM API keys configured -- ensemble will run on "
                  "Linear Regression + LSTM only.")
        return []

    results = []
    # NOT a `with` block: its __exit__ waits for every thread, which made the panel-level
    # ceiling meaningless (a provider sleeping 6s held a 1s ceiling for 6s -- verified).
    executor = ThreadPoolExecutor(max_workers=len(configured))
    future_to_client = {executor.submit(c.get_prediction, trend_summary): c for c in configured}
    pending = set(future_to_client)
    deadline = time.monotonic() + config.LLM_PANEL_TIMEOUT_SECONDS
    try:
        while pending:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            done, pending = wait(pending, timeout=left, return_when=FIRST_COMPLETED)
            for future in done:
                client = future_to_client[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = client._error_result(str(e))
                results.append(result)
                _log(result, verbose)
        for future in pending:   # abandoned, never awaited
            client = future_to_client[future]
            result = client._error_result(
                f"Timed out waiting for a response (panel-level {config.LLM_PANEL_TIMEOUT_SECONDS}s ceiling).")
            results.append(result)
            _log(result, verbose)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results


def _log(result, verbose):
    if not verbose:
        return
    if result["error"]:
        print(f"[panel] {result['provider']}: ERROR - {result['error']}")
    else:
        print(f"[panel] {result['provider']}: {result['direction']} "
              f"({result['estimated_pct_change']:+.2f}%, confidence {result['confidence']:.2f})")
