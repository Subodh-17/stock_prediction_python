"""ChatGPT (OpenAI) client for the AI prediction panel."""

from .base_client import BaseLLMClient
import config


class OpenAIClient(BaseLLMClient):
    provider_name = "ChatGPT"

    def __init__(self):
        super().__init__(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)

    def _call_api(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, timeout=30.0)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content
