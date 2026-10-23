"""Configurable, cost-aware model selection."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.core.catalog import MODEL_REGISTRY, ModelConfig, QualityTier, paid_providers_enabled
from app.core.complexity import ComplexityResult
from app.core.config import CONFIG_DIR
from app.core.costing import estimate_cost_usd

DEFAULT_RULES_PATH = CONFIG_DIR / "routing.yaml"
COMPLEXITY_LABELS = frozenset({"simple", "moderate", "complex"})


@dataclass(frozen=True)
class RouteDecision:
    model: ModelConfig
    reason: str
    requested_tier: QualityTier
    selected_tier: QualityTier
    estimated_cost_usd: float
    used_fallback: bool


def routing_rules_path() -> Path:
    """Where live routing rules are read from and written to (``LLM_ROUTING_RULES_PATH``)."""
    configured = os.getenv("LLM_ROUTING_RULES_PATH", "").strip()
    return Path(configured) if configured else DEFAULT_RULES_PATH


def load_routing_rules(path: Path | None = None) -> dict:
    if path is None:
        path = routing_rules_path()
        if not path.exists():  # e.g. a fresh Docker volume: start from the packaged defaults
            path = DEFAULT_RULES_PATH
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict):
        raise TypeError("Routing configuration must contain a mapping.")
    if set(rules.get("class_to_quality_tier", {})) != COMPLEXITY_LABELS:
        raise ValueError("Routing rules must map simple, moderate, and complex labels.")
    return rules


def _validated_candidates(tier_to_models: dict[str, list[str]], rules: dict) -> dict:
    if set(tier_to_models) != COMPLEXITY_LABELS:
        raise ValueError("tier_to_models must contain simple, moderate, and complex.")
    validated: dict[str, list[str]] = {}
    for tier, model_ids in tier_to_models.items():
        if not model_ids:
            raise ValueError(f"At least one candidate is required for the {tier} tier.")
        if len(set(model_ids)) != len(model_ids):
            raise ValueError(f"The {tier} tier contains duplicate model IDs.")
        minimum = QualityTier(rules["class_to_quality_tier"][tier])
        for model_id in model_ids:
            model = MODEL_REGISTRY.get(model_id)
            if model is None:
                raise ValueError(f"Unknown model_id in {tier} tier: {model_id}.")
            if model.quality_tier.rank < minimum.rank:
                raise ValueError(
                    f"Model {model_id} is below the configured minimum quality for {tier}."
                )
        validated[tier] = list(model_ids)
    return validated


def _write_atomically(path: Path, rules: dict) -> None:
    """Write via a temp file + rename so a concurrent reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            yaml.safe_dump(rules, file, sort_keys=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def update_tier_to_models(tier_to_models: dict[str, list[str]]) -> dict:
    """Validate and persist candidate mappings; the router re-reads them on every request."""
    rules = load_routing_rules()
    rules["model_candidates_by_tier"] = _validated_candidates(tier_to_models, rules)
    _write_atomically(routing_rules_path(), rules)
    return rules


def choose_model(
    complexity: ComplexityResult,
    minimum_quality: QualityTier = QualityTier.LOW,
    *,
    registry: dict[str, ModelConfig] | None = None,
    rules: dict | None = None,
    allow_paid: bool | None = None,
    prompt: str = "",
) -> RouteDecision:
    """Pick the cheapest eligible model for the prompt's complexity tier.

    Configured candidates for the tier win; if none is eligible (for example hosted models
    are disabled) the router falls back through ``fallback_tiers`` to the best tier available.
    """
    registry = MODEL_REGISTRY if registry is None else registry
    rules = load_routing_rules() if rules is None else rules
    allow_paid = paid_providers_enabled() if allow_paid is None else allow_paid
    label = complexity.label
    requested_tier = max(
        QualityTier(rules["class_to_quality_tier"][label]), minimum_quality, key=lambda t: t.rank
    )

    eligible = [
        model
        for model in registry.values()
        if model.quality_tier.rank >= minimum_quality.rank and model.is_available(allow_paid)
    ]
    preferred = rules.get("model_candidates_by_tier", {}).get(label, [])
    candidates = [model for model in eligible if model.model_id in preferred]
    used_fallback = not candidates
    if used_fallback:
        for tier in rules["fallback_tiers"].get(requested_tier.value, [requested_tier.value]):
            candidates = [model for model in eligible if model.quality_tier.value == tier]
            if candidates:
                break
    if not candidates:
        available = ", ".join(sorted(model.model_id for model in eligible)) or "none"
        raise ValueError(
            f"No model can meet requested tier '{requested_tier.value}'. "
            f"Available eligible models: {available}."
        )

    input_tokens = max(1, (len(prompt) + 3) // 4)  # rough chars-to-tokens estimate pre-call
    output_tokens = int(rules["estimated_output_tokens"][label])

    def route_cost(model: ModelConfig) -> float:
        return estimate_cost_usd(
            model,
            input_tokens,
            output_tokens,
            input_weight=float(rules.get("input_cost_weight", 1.0)),
            output_weight=float(rules.get("output_cost_weight", 1.0)),
        )

    model = min(candidates, key=lambda item: (route_cost(item), item.model_id))
    scope = (
        f"model in {model.quality_tier.value} quality tier"
        if used_fallback
        else f"model among configured {label}-tier candidates"
    )
    reason = (
        f"{label} prompt (classifier confidence {complexity.confidence:.0%}) "
        f"requested {requested_tier.value}; selected the lowest estimated-cost eligible {scope}."
    )
    if used_fallback:
        reason += " Requested tier was unavailable under the current provider settings."
    return RouteDecision(
        model=model,
        reason=reason,
        requested_tier=requested_tier,
        selected_tier=model.quality_tier,
        estimated_cost_usd=route_cost(model),
        used_fallback=used_fallback,
    )
