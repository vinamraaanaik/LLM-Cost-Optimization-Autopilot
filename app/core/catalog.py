"""Model registry: provider metadata, per-token prices, and runtime availability."""

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.core.config import DATA_DIR, env_flag

# Hosted providers are only usable when their API key is present in the environment.
CREDENTIAL_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class QualityTier(StrEnum):
    """Quality levels declared in ascending order; a higher rank is a more capable model."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return tuple(type(self)).index(self)


def paid_providers_enabled() -> bool:
    """Hosted providers cost real money, so they require an explicit opt-in."""
    return env_flag("LLM_ALLOW_PAID_PROVIDERS")


@dataclass(frozen=True)
class ModelConfig:
    """Provider/model metadata. Costs are USD per token."""

    provider_name: str
    model_id: str
    cost_per_input_token: float
    cost_per_output_token: float
    quality_tier: QualityTier
    local: bool

    def is_available(self, allow_paid: bool | None = None) -> bool:
        """Local models are always usable; hosted ones need the opt-in and an API key."""
        if self.local:
            return True
        if allow_paid is None:
            allow_paid = paid_providers_enabled()
        key_name = CREDENTIAL_ENV.get(self.provider_name)
        return allow_paid and bool(key_name and os.getenv(key_name))


def load_model_registry(
    models_path: Path = DATA_DIR / "models.json",
    pricing_path: Path = DATA_DIR / "pricing.json",
) -> dict[str, ModelConfig]:
    models = json.loads(models_path.read_text(encoding="utf-8"))
    pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
    if pricing.get("currency") != "USD" or pricing.get("unit") != "per_million_tokens":
        raise ValueError("pricing.json must define USD rates per million tokens.")

    registry: dict[str, ModelConfig] = {}
    for model in models:
        model_id = model["model_id"]
        rates = pricing["models"][model_id]
        registry[model_id] = ModelConfig(
            provider_name=model["provider_name"],
            model_id=model_id,
            cost_per_input_token=float(rates["input"]) / 1_000_000,
            cost_per_output_token=float(rates["output"]) / 1_000_000,
            quality_tier=QualityTier(model["quality_tier"]),
            local=bool(model["local"]),
        )
    return registry


MODEL_REGISTRY = load_model_registry()
