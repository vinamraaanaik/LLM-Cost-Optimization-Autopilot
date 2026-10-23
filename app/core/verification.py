"""Quality verification: compare a response against a higher-tier model and escalate on failure."""

import json
from collections.abc import Awaitable, Callable

from app.core.catalog import MODEL_REGISTRY, ModelConfig
from app.core.complexity import extract_features
from app.core.config import DEFAULT_MAX_OUTPUT_TOKENS
from app.core.database import Database
from app.core.quality import compare_outputs, infer_request_type
from app.providers.base import Response

Sender = Callable[..., Awaitable[Response]]

# A routing failure teaches the classifier that this prompt needed the next tier up.
NEXT_TIER = {"simple": "moderate", "moderate": "complex", "complex": "complex"}


def select_verifier_model(
    original_model: ModelConfig, registry: dict[str, ModelConfig] | None = None
) -> ModelConfig | None:
    """Cheapest available model in the highest tier above the one that answered."""
    registry = MODEL_REGISTRY if registry is None else registry
    stronger = [
        model
        for model in registry.values()
        if model.quality_tier.rank > original_model.quality_tier.rank and model.is_available()
    ]
    if not stronger:
        return None
    top_rank = max(model.quality_tier.rank for model in stronger)
    return min(
        (model for model in stronger if model.quality_tier.rank == top_rank),
        key=lambda m: (m.cost_per_input_token + m.cost_per_output_token, m.model_id),
    )


async def verify_and_escalate(
    *,
    database: Database,
    request_id: str,
    prompt: str,
    request_type: str,
    complexity: str,
    original_model: ModelConfig,
    original_response: Response,
    send: Sender,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict:
    """Compare with a higher-tier answer and replace a failed response with it.

    The reference answer was generated from the same prompt, so on failure it simply becomes
    the escalated answer; no second escalation call is needed.
    """
    request_type = infer_request_type(prompt, request_type)
    verifier = select_verifier_model(original_model)
    if verifier is None:
        reason = "No higher-tier verifier is available under current provider settings."
        database.update_verification(
            request_id, status="skipped", request_type=request_type, reason=reason
        )
        return {"status": "skipped", "reason": reason}

    database.update_verification(
        request_id,
        status="running",
        request_type=request_type,
        verifier_model_id=verifier.model_id,
        reason="Higher-tier reference response is being checked.",
    )
    try:
        reference = await send(prompt, verifier, max_output_tokens=max_output_tokens)
    except Exception as error:  # any verifier failure must end in a terminal status
        reason = f"Verifier request failed ({type(error).__name__})."
        database.update_verification(
            request_id,
            status="error",
            request_type=request_type,
            verifier_model_id=verifier.model_id,
            reason=reason,
        )
        return {"status": "error", "reason": reason}

    assessment = compare_outputs(
        prompt, original_response.output_text, reference.output_text, request_type
    )
    outcome = {
        "request_type": assessment.request_type,
        "verifier_model_id": verifier.model_id,
        "quality_score": assessment.score,
        "quality_threshold": assessment.threshold,
        "quality_gap": assessment.gap,
        "reason": assessment.reason,
        "verifier_response": reference,
    }
    if assessment.failed:
        database.record_routing_failure(
            {
                "request_id": request_id,
                "prompt": prompt,
                "complexity": complexity,
                "target_complexity": NEXT_TIER[complexity],
                "request_type": assessment.request_type,
                "original_model_id": original_model.model_id,
                "verifier_model_id": verifier.model_id,
                "quality_score": assessment.score,
                "quality_threshold": assessment.threshold,
                "quality_gap": assessment.gap,
                "reason": assessment.reason,
                "features_json": json.dumps(extract_features(prompt), ensure_ascii=False),
            }
        )
        database.update_verification(
            request_id,
            status="escalated",
            verifier_model=verifier,
            original_cost_usd=original_response.cost_usd,
            escalated=True,
            **outcome,
        )
    else:
        database.update_verification(request_id, status="verified", **outcome)

    return {
        "status": "escalated" if assessment.failed else "verified",
        "request_type": assessment.request_type,
        "quality_score": assessment.score,
        "quality_threshold": assessment.threshold,
        "quality_gap": assessment.gap,
        "reason": assessment.reason,
        "verifier_model_id": verifier.model_id,
        "escalated": assessment.failed,
        "escalation_model_id": verifier.model_id if assessment.failed else None,
        "escalation_cost_delta_usd": (
            round(reference.cost_usd - original_response.cost_usd, 10) if assessment.failed else 0.0
        ),
    }
