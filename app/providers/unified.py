from contextlib import AsyncExitStack
from time import perf_counter

import httpx

from app.core.catalog import ModelConfig, paid_providers_enabled
from app.core.config import DEFAULT_MAX_OUTPUT_TOKENS
from app.core.costing import estimate_cost_usd
from app.providers.anthropic import AnthropicAdapter
from app.providers.base import ProviderAdapter, ProviderConfigurationError, Response
from app.providers.ollama import OllamaAdapter
from app.providers.openai import OpenAIAdapter

PROVIDERS: dict[str, ProviderAdapter] = {
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "ollama": OllamaAdapter(),
}
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


async def send_request(
    prompt: str,
    model_config: ModelConfig,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    client: httpx.AsyncClient | None = None,
) -> Response:
    """Call a registered provider and normalize output, usage, latency, and cost."""
    if not model_config.local and not paid_providers_enabled():
        raise ProviderConfigurationError(
            "Hosted providers are disabled. Set LLM_ALLOW_PAID_PROVIDERS=true to opt in."
        )
    adapter = PROVIDERS.get(model_config.provider_name)
    if adapter is None:
        raise ProviderConfigurationError(
            f"No adapter is registered for provider '{model_config.provider_name}'."
        )

    started = perf_counter()
    async with AsyncExitStack() as stack:
        if client is None:  # callers may share one client across many calls
            client = await stack.enter_async_context(httpx.AsyncClient(timeout=_TIMEOUT))
        result = await adapter.complete(prompt, model_config, max_output_tokens, client)

    return Response(
        output_text=result.output_text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=round((perf_counter() - started) * 1_000, 3),
        cost_usd=estimate_cost_usd(model_config, result.input_tokens, result.output_tokens),
        model_id=model_config.model_id,
    )
