import pytest

from app.core.catalog import MODEL_REGISTRY, ModelConfig, QualityTier
from app.core.complexity import ComplexityResult, classify_prompt
from app.core.router import choose_model, load_routing_rules, update_tier_to_models

RULES = {
    "class_to_quality_tier": {"simple": "low", "moderate": "medium", "complex": "high"},
    "fallback_tiers": {
        "low": ["low"],
        "medium": ["medium", "low"],
        "high": ["high", "medium", "low"],
    },
    "estimated_output_tokens": {"simple": 96, "moderate": 192, "complex": 384},
    "input_cost_weight": 1.0,
    "output_cost_weight": 4.0,
}


def test_default_routing_rules_load_independently_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rules = load_routing_rules()
    assert rules["class_to_quality_tier"] == RULES["class_to_quality_tier"]
    assert "llama3.2:1b" in rules["model_candidates_by_tier"]["simple"]


def test_missing_configured_rules_file_falls_back_to_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_ROUTING_RULES_PATH", str(tmp_path / "not-created-yet.yaml"))
    assert "model_candidates_by_tier" in load_routing_rules()


def test_incomplete_rules_are_rejected(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("class_to_quality_tier:\n  simple: low\n", encoding="utf-8")
    with pytest.raises(ValueError, match="simple, moderate, and complex"):
        load_routing_rules(path)


def test_router_uses_local_provider_by_default():
    local = ModelConfig("ollama", "local-small", 0.0, 0.0, QualityTier.LOW, True)
    paid = ModelConfig("openai", "paid-high", 0.0001, 0.0002, QualityTier.HIGH, False)
    result = choose_model(
        classify_prompt("Design a multi-stage evaluation plan with tradeoffs."),
        registry={local.model_id: local, paid.model_id: paid},
        rules=RULES,
        allow_paid=False,
        prompt="design a plan",
    )
    assert result.model.model_id == local.model_id


def test_simple_tier_picks_the_cheapest_eligible_candidate(monkeypatch):
    monkeypatch.setenv("LLM_ALLOW_PAID_PROVIDERS", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    rules = load_routing_rules()
    candidates = rules["model_candidates_by_tier"]["simple"]
    assert {"llama3.2:1b", "claude-haiku-5-5"} <= set(candidates)
    selected = choose_model(
        ComplexityResult("simple", 0.9, 0.15),
        registry={model_id: MODEL_REGISTRY[model_id] for model_id in candidates},
        rules=rules,
        allow_paid=True,
        prompt="What time is the meeting?",
    )
    assert selected.model.model_id == "llama3.2:1b"  # free beats Haiku


def test_complex_tier_falls_back_to_local_when_paid_models_are_disabled():
    decision = choose_model(
        ComplexityResult("complex", 0.9, 0.85),
        allow_paid=False,
        prompt="Analyze the trade-offs and recommend an approach.",
    )
    assert decision.used_fallback
    assert decision.model.local
    assert decision.model.quality_tier == QualityTier.MEDIUM
    assert "unavailable" in decision.reason


def test_minimum_quality_raises_the_requested_tier():
    decision = choose_model(
        ComplexityResult("simple", 0.9, 0.15),
        minimum_quality=QualityTier.MEDIUM,
        allow_paid=False,
        prompt="hi",
    )
    assert decision.requested_tier == QualityTier.MEDIUM
    assert decision.model.quality_tier.rank >= QualityTier.MEDIUM.rank


def test_router_fails_clearly_when_nothing_is_eligible():
    paid = ModelConfig("openai", "paid-only", 1e-6, 1e-6, QualityTier.HIGH, False)
    with pytest.raises(ValueError, match="No model can meet requested tier"):
        choose_model(
            ComplexityResult("complex", 0.9, 0.85),
            registry={paid.model_id: paid},
            rules=RULES,
            allow_paid=False,
        )


def test_router_estimates_price_with_input_and_output_weights():
    model = ModelConfig("ollama", "local-medium", 0.000001, 0.000002, QualityTier.MEDIUM, True)
    decision = choose_model(
        ComplexityResult(label="moderate", confidence=0.8, score=0.5),
        registry={model.model_id: model},
        rules=RULES,
        allow_paid=False,
        prompt="abcd",
    )
    assert decision.estimated_cost_usd == 0.001537


VALID_MAPPING = {
    "simple": ["llama3.2:1b"],
    "moderate": ["llama3.1:8b"],
    "complex": ["claude-sonnet-5"],
}


def test_update_persists_and_is_read_back(monkeypatch, tmp_path):
    path = tmp_path / "live-rules.yaml"
    monkeypatch.setenv("LLM_ROUTING_RULES_PATH", str(path))
    update_tier_to_models(VALID_MAPPING)
    assert path.exists()
    assert load_routing_rules()["model_candidates_by_tier"] == VALID_MAPPING
    assert list(tmp_path.glob("*.tmp")) == []  # the atomic-write temp file is cleaned up


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({**VALID_MAPPING, "complex": ["llama3.2:1b"]}, "below the configured minimum"),
        ({**VALID_MAPPING, "simple": ["no-such-model"]}, "Unknown model_id"),
        ({**VALID_MAPPING, "simple": ["llama3.2:1b", "llama3.2:1b"]}, "duplicate"),
        ({**VALID_MAPPING, "moderate": []}, "At least one candidate"),
        ({"simple": ["llama3.2:1b"]}, "must contain simple, moderate, and complex"),
    ],
)
def test_update_rejects_invalid_mappings_without_writing(monkeypatch, tmp_path, mapping, message):
    path = tmp_path / "live-rules.yaml"
    monkeypatch.setenv("LLM_ROUTING_RULES_PATH", str(path))
    with pytest.raises(ValueError, match=message):
        update_tier_to_models(mapping)
    assert not path.exists()
