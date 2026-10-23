import httpx
import pytest

from app.core.catalog import MODEL_REGISTRY
from app.providers import unified
from app.providers.base import ProviderConfigurationError, Response
from app.providers.unified import send_request
from tests.helpers import run

PAYLOADS = {
    "gpt-4o-mini": lambda i, o: {
        "choices": [{"message": {"content": "mocked response"}}],
        "usage": {"prompt_tokens": i, "completion_tokens": o},
    },
    "claude-haiku-5-5": lambda i, o: {
        "content": [{"type": "text", "text": "mocked response"}],
        "usage": {"input_tokens": i, "output_tokens": o},
    },
    "llama3.2:1b": lambda i, o: {
        "message": {"content": "mocked response"},
        "prompt_eval_count": i,
        "eval_count": o,
    },
}


@pytest.mark.parametrize(
    ("model_id", "expected_input", "expected_output"),
    [("gpt-4o-mini", 12, 4), ("claude-haiku-5-5", 9, 3), ("llama3.2:1b", 7, 2)],
)
def test_adapters_normalize_provider_responses(
    monkeypatch, model_id, expected_input, expected_output
):
    monkeypatch.setenv("LLM_ALLOW_PAID_PROVIDERS", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def handler(request):
        return httpx.Response(200, json=PAYLOADS[model_id](expected_input, expected_output))

    async def call():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await send_request("hello", MODEL_REGISTRY[model_id], client=client)

    result = run(call())
    assert result.output_text == "mocked response"
    assert (result.input_tokens, result.output_tokens) == (expected_input, expected_output)
    assert result.total_tokens == expected_input + expected_output
    assert result.cost_usd >= 0
    assert result.model_id == model_id


def test_hosted_call_is_blocked_without_opt_in():
    with pytest.raises(ProviderConfigurationError, match="disabled"):
        run(send_request("hello", MODEL_REGISTRY["gpt-4o-mini"]))


def test_hosted_call_without_a_key_fails_in_the_adapter(monkeypatch):
    monkeypatch.setenv("LLM_ALLOW_PAID_PROVIDERS", "true")

    async def call():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: None)) as client:
            return await send_request("hello", MODEL_REGISTRY["gpt-4o-mini"], client=client)

    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        run(call())


def test_unregistered_provider_is_reported(monkeypatch):
    monkeypatch.setattr(unified, "PROVIDERS", {})
    with pytest.raises(ProviderConfigurationError, match="No adapter is registered"):
        run(send_request("hello", MODEL_REGISTRY["llama3.2:1b"]))


def test_response_total_tokens_is_derived():
    response = Response("x", 3, 4, 1.0, 0.0, "m")
    assert response.total_tokens == 7
