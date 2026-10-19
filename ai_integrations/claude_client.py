"""Claude (Anthropic) client for the AI prediction panel."""

from .base_client import BaseLLMClient
import config


class ClaudeClient(BaseLLMClient):
    provider_name = "Claude"

    def __init__(self):
        super().__init__(api_key=config.ANTHROPIC_API_KEY, model=config.ANTHROPIC_MODEL)

    def _call_api(self, prompt: str) -> str:
        import anthropic
        # Explicit timeout -- without it, a hung request could block this
        # whole call for the SDK's much longer default, which in turn blocks
        # the entire panel (all 5 providers) since query_llm_panel waits on
        # every submitted future before returning.
        client = anthropic.Anthropic(api_key=self.api_key, timeout=30.0)
        response = client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
