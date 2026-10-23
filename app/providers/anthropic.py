import os

import httpx

from app.core.catalog import ModelConfig
from app.providers.base import ProviderConfigurationError, ProviderResult


class AnthropicAdapter:
    async def complete(
        self,
        prompt: str,
        model_config: ModelConfig,
        max_output_tokens: int,
        client: httpx.AsyncClient,
    ) -> ProviderResult:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderConfigurationError("ANTHROPIC_API_KEY is not set.")

        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": model_config.model_id,
                "max_tokens": max_output_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()
        usage = data["usage"]
        text = "".join(
            block.get("text", "") for block in data["content"] if block.get("type") == "text"
        )
        return ProviderResult(
            output_text=text,
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage["output_tokens"]),
        )
