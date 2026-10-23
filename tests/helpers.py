"""Shared test helpers (imported as ``tests.helpers``)."""

import asyncio

from app.providers.base import Response

EXTRACTION_PROMPT = "Extract name and city from the note as JSON."
SMALL_ANSWER = '{"name":"Ada"}'  # what a weak model returns: the city is missing
FULL_ANSWER = '{"name":"Ada","city":"London"}'  # what the stronger verifier returns


def run(coroutine):
    """Run a coroutine to completion without needing an async pytest plugin."""
    return asyncio.run(coroutine)


def request_row(**overrides) -> dict:
    """A minimal valid row for ``Database.record_request``."""
    return {
        "id": "r1",
        "prompt": "hello",
        "model_id": "llama3.2:1b",
        "provider_name": "ollama",
        "complexity": "simple",
        "confidence": 0.8,
        "route_reason": "test route",
        "used_fallback": 0,
        "output_text": "hi",
        "input_tokens": 2,
        "output_tokens": 3,
        "latency_ms": 20.0,
        "cost_usd": 0.0,
        **overrides,
    }


def response(
    model_id: str = "llama3.2:1b",
    output_text: str = "ok",
    *,
    input_tokens: int = 8,
    output_tokens: int = 7,
    latency_ms: float = 5.0,
    cost_usd: float = 0.0,
) -> Response:
    return Response(
        output_text=output_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        model_id=model_id,
    )


def fake_send(outputs: dict[str, str] | None = None, costs: dict[str, float] | None = None):
    """Async stand-in for ``send_request`` with canned per-model outputs; records its calls."""
    outputs, costs = outputs or {}, costs or {}

    async def send(prompt, model_config, **_):
        send.calls.append((prompt, model_config.model_id))
        return response(
            model_config.model_id,
            outputs.get(model_config.model_id, "ok"),
            cost_usd=costs.get(model_config.model_id, 0.0),
        )

    send.calls = []
    return send


def escalating_send():
    """The weak model misses a field; the verifier (llama3.1:8b) returns the full answer."""
    return fake_send(
        outputs={"llama3.2:1b": SMALL_ANSWER, "llama3.1:8b": FULL_ANSWER},
        costs={"llama3.2:1b": 0.0001, "llama3.1:8b": 0.0005},
    )
