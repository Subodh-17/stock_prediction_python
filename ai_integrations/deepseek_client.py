"""
DeepSeek client for the AI prediction panel.
DeepSeek also exposes an OpenAI-compatible endpoint.
"""

from .base_client import BaseLLMClient
import config


class DeepSeekClient(BaseLLMClient):
    provider_name = "DeepSeek"
    BASE_URL = "https://api.deepseek.com"

    def __init__(self):
        super().__init__(api_key=config.DEEPSEEK_API_KEY, model=config.DEEPSEEK_MODEL)

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
