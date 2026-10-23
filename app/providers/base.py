from dataclasses import dataclass
from typing import Protocol

import httpx

from app.core.catalog import ModelConfig


@dataclass(frozen=True)
class ProviderResult:
    """What an adapter returns: text and provider-reported token usage."""

    output_text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Response:
    """Normalized result of one model call, including latency and estimated cost."""

    output_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model_id: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ProviderAdapter(Protocol):
    async def complete(
        self,
        prompt: str,
        model_config: ModelConfig,
        max_output_tokens: int,
        client: httpx.AsyncClient,
    ) -> ProviderResult:
        """Call a provider and return normalized text and token usage."""


class ProviderConfigurationError(RuntimeError):
    """Raised when required provider configuration is unavailable."""
