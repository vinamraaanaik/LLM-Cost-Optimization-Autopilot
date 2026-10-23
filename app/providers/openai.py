import os

import httpx

from app.core.catalog import ModelConfig
from app.providers.base import ProviderConfigurationError, ProviderResult


class OpenAIAdapter:
    async def complete(
        self,
        prompt: str,
        model_config: ModelConfig,
        max_output_tokens: int,
        client: httpx.AsyncClient,
    ) -> ProviderResult:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is not set.")

        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_config.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_output_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        usage = data["usage"]
        return ProviderResult(
            output_text=data["choices"][0]["message"]["content"] or "",
            input_tokens=int(usage["prompt_tokens"]),
            output_tokens=int(usage["completion_tokens"]),
        )
