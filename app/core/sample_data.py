"""Clearly synthetic request rows so the dashboard has something to draw on an empty database."""

from datetime import UTC, date, datetime, timedelta

from app.core.catalog import MODEL_REGISTRY, ModelConfig
from app.core.costing import estimate_cost_usd


def _row(
    registry: dict[str, ModelConfig],
    day: date,
    clock: str,
    *,
    model_id: str,
    complexity: str,
    input_tokens: int,
    output_tokens: int,
    quality_score: float | None = None,
    escalated: bool = False,
    verifier_id: str | None = None,
) -> dict:
    """One request, plus the verifier call that would have been made for it."""
    calls = [model_id] + ([verifier_id] if verifier_id else [])
    final_model_id = verifier_id if escalated else model_id
    return {
        "id": f"sample-{day.isoformat()}-{clock[:2]}-{model_id}",
        "created_at": datetime.fromisoformat(f"{day.isoformat()}T{clock}:00+00:00").isoformat(),
        "prompt_hash": "sample data",
        "original_model_id": model_id,
        "model_id": final_model_id,
        "complexity": complexity,
        "latency_ms": 780 if final_model_id.startswith("llama") else 1450,
        "cost_usd": sum(
            estimate_cost_usd(registry[called], input_tokens, output_tokens) for called in calls
        ),
        "quality_score": quality_score,
        "escalated": int(escalated),
        "input_tokens": input_tokens * len(calls),
        "output_tokens": output_tokens * len(calls),
        "verification_status": "completed" if verifier_id else "skipped",
    }


def sample_requests(
    now: datetime | None = None, registry: dict[str, ModelConfig] | None = None
) -> list[dict]:
    """Three requests a day for 14 days; every 5th day the moderate request escalates."""
    registry = MODEL_REGISTRY if registry is None else registry
    today = (now or datetime.now(UTC)).date()
    rows = []
    for offset in range(13, -1, -1):
        day = today - timedelta(days=offset)
        moderate_fails = offset % 5 == 0
        rows += [
            _row(
                registry,
                day,
                "09:15",
                model_id="llama3.2:1b",
                complexity="simple",
                input_tokens=140,
                output_tokens=55,
                quality_score=0.98 if offset % 4 else 0.91,
                verifier_id="llama3.1:8b",
            ),
            _row(
                registry,
                day,
                "13:30",
                model_id="gpt-4o-mini",
                complexity="moderate",
                input_tokens=260,
                output_tokens=120,
                quality_score=0.82 if moderate_fails else 0.91,
                escalated=moderate_fails,
                verifier_id="claude-sonnet-5",
            ),
            _row(
                registry,
                day,
                "19:45",
                model_id="claude-sonnet-5",
                complexity="complex",
                input_tokens=620,
                output_tokens=350,
            ),
        ]
    return rows
