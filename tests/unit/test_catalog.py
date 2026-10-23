import pytest

from app.core.catalog import MODEL_REGISTRY, ModelConfig, QualityTier
from app.core.costing import estimate_cost_usd, savings_vs_baseline


def hosted(provider: str = "openai") -> ModelConfig:
    return ModelConfig(provider, "hosted-model", 1e-6, 2e-6, QualityTier.HIGH, False)


def test_quality_tiers_are_ordered():
    assert QualityTier.LOW.rank < QualityTier.MEDIUM.rank < QualityTier.HIGH.rank


def test_registry_prices_are_per_token_and_local_models_are_free():
    assert MODEL_REGISTRY["gpt-4o"].cost_per_input_token == pytest.approx(2.5e-6)
    assert MODEL_REGISTRY["gpt-4o"].cost_per_output_token == pytest.approx(10e-6)
    local = [model for model in MODEL_REGISTRY.values() if model.local]
    assert local
    assert all(m.cost_per_input_token == 0 and m.cost_per_output_token == 0 for m in local)


def test_local_models_are_always_available():
    assert MODEL_REGISTRY["llama3.2:1b"].is_available()
    assert MODEL_REGISTRY["llama3.2:1b"].is_available(allow_paid=False)


def test_hosted_models_need_opt_in_and_a_key(monkeypatch):
    model = hosted()
    assert not model.is_available()
    monkeypatch.setenv("LLM_ALLOW_PAID_PROVIDERS", "true")
    assert not model.is_available()  # opted in, but no key
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert model.is_available()
    assert not model.is_available(allow_paid=False)  # an explicit argument overrides the env


def test_keys_are_matched_per_provider(monkeypatch):
    monkeypatch.setenv("LLM_ALLOW_PAID_PROVIDERS", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert hosted("anthropic").is_available()
    assert not hosted("openai").is_available()
    assert not hosted("unknown-provider").is_available()


def test_cost_estimate():
    model = ModelConfig("example", "x", 0.000001, 0.000002, QualityTier.LOW, True)
    assert estimate_cost_usd(model, 100, 50) == 0.0002


def test_cost_estimate_rejects_negative_tokens():
    with pytest.raises(ValueError, match="negative"):
        estimate_cost_usd(hosted(), -1, 0)


def test_savings_against_baseline():
    assert savings_vs_baseline(actual_usd=2.0, baseline_usd=10.0) == (8.0, 80.0)
    assert savings_vs_baseline(actual_usd=0.0, baseline_usd=0.0) == (0.0, None)
    savings, percent = savings_vs_baseline(actual_usd=12.0, baseline_usd=10.0)
    assert savings == -2.0 and percent == pytest.approx(-20.0)
