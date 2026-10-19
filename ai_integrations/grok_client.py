"""
Grok (xAI) client for the AI prediction panel.
xAI exposes an OpenAI-compatible endpoint, so we reuse the `openai` SDK
pointed at their base URL rather than needing a separate package.
"""

from .base_client import BaseLLMClient
import config


class GrokClient(BaseLLMClient):
    provider_name = "Grok"
    BASE_URL = "https://api.x.ai/v1"

    def __init__(self):
        super().__init__(api_key=config.XAI_API_KEY, model=config.XAI_MODEL)

    def _call_api(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL, timeout=30.0)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content
