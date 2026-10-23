import os

import httpx

from app.core.catalog import ModelConfig
from app.providers.base import ProviderResult


class OllamaAdapter:
    async def complete(
        self,
        prompt: str,
        model_config: ModelConfig,
        max_output_tokens: int,
        client: httpx.AsyncClient,
    ) -> ProviderResult:
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        response = await client.post(
            f"{host}/api/chat",
            json={
                "model": model_config.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": max_output_tokens},
            },
        )
        response.raise_for_status()
        data = response.json()
        return ProviderResult(
            output_text=data["message"]["content"],
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
        )
