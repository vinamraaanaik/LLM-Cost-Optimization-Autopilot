from app.core import verification
from app.core.catalog import MODEL_REGISTRY
from app.core.complexity import training_samples
from app.core.verification import select_verifier_model, verify_and_escalate
from tests.helpers import (
    EXTRACTION_PROMPT,
    FULL_ANSWER,
    SMALL_ANSWER,
    escalating_send,
    fake_send,
    request_row,
    response,
    run,
)

WEAK = MODEL_REGISTRY["llama3.2:1b"]
STRONG = MODEL_REGISTRY["llama3.1:8b"]


def record_pending_extraction(database, output=SMALL_ANSWER):
    database.record_request(
        request_row(
            prompt=EXTRACTION_PROMPT,
            output_text=output,
            input_tokens=5,
            output_tokens=4,
            cost_usd=0.0001,
            request_type="extraction",
            verification_status="pending",
        )
    )


def verify(database, send, output=SMALL_ANSWER):
    return run(
        verify_and_escalate(
            database=database,
            request_id="r1",
            prompt=EXTRACTION_PROMPT,
            request_type="extraction",
            complexity="simple",
            original_model=WEAK,
            original_response=response(
                WEAK.model_id, output, input_tokens=5, output_tokens=4, cost_usd=0.0001
            ),
            send=send,
        )
    )


def test_local_verifier_is_the_next_tier_up():
    assert select_verifier_model(WEAK).model_id == "llama3.1:8b"


def test_hosted_verifier_is_chosen_only_with_opt_in_and_key(monkeypatch):
    assert select_verifier_model(STRONG) is None  # nothing above local-medium is usable
    monkeypatch.setenv("LLM_ALLOW_PAID_PROVIDERS", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert select_verifier_model(STRONG).model_id == "claude-sonnet-5"  # cheapest high-tier


def test_matching_response_is_verified_without_escalation(database, monkeypatch):
    monkeypatch.setattr(verification, "select_verifier_model", lambda _model: STRONG)
    record_pending_extraction(database, output=FULL_ANSWER)
    send = fake_send({"llama3.1:8b": FULL_ANSWER}, {"llama3.1:8b": 0.0005})

    result = verify(database, send, output=FULL_ANSWER)

    stored = database.request_by_id("r1")
    assert result["status"] == "verified"
    assert stored["verification_status"] == "verified"
    assert not stored["escalated"]
    assert stored["output_text"] == FULL_ANSWER
    assert stored["cost_usd"] == 0.0006  # the verifier's cost is added to the request
    assert database.training_failures() == []


def test_failed_check_escalates_and_feeds_the_classifier(database, monkeypatch):
    monkeypatch.setattr(verification, "select_verifier_model", lambda _model: STRONG)
    record_pending_extraction(database)

    result = verify(database, escalating_send())

    stored = database.request_by_id("r1")
    assert result["status"] == "escalated"
    assert result["escalation_cost_delta_usd"] == 0.0004
    assert stored["verification_status"] == "escalated"
    assert stored["original_model_id"] == WEAK.model_id
    assert stored["model_id"] == STRONG.model_id
    assert stored["initial_output_text"] == SMALL_ANSWER  # the first answer is kept for audit
    assert stored["output_text"] == FULL_ANSWER
    assert stored["escalation_model_id"] == STRONG.model_id
    assert stored["escalation_cost_delta_usd"] == 0.0004
    assert abs(stored["cost_usd"] - 0.0006) < 1e-12

    failure = database.recent_routing_failures()[0]
    assert failure["original_model_id"] == WEAK.model_id
    assert failure["escalation_model_id"] == STRONG.model_id
    assert failure["quality_gap"] > 0

    # simple -> moderate: the next retrain sees this prompt labeled with the tier it needed.
    samples = training_samples(database.training_failures())
    assert samples[-1] == {
        "text": EXTRACTION_PROMPT,
        "label": "moderate",
        "source": "routing_failure",
    }


def test_verification_is_skipped_when_no_stronger_model_is_available(database, monkeypatch):
    monkeypatch.setattr(verification, "select_verifier_model", lambda _model: None)
    record_pending_extraction(database)
    send = fake_send()

    result = verify(database, send)

    assert result["status"] == "skipped"
    assert send.calls == []
    assert database.request_by_id("r1")["verification_status"] == "skipped"


def test_verifier_failure_ends_in_a_terminal_error_status(database, monkeypatch):
    monkeypatch.setattr(verification, "select_verifier_model", lambda _model: STRONG)
    record_pending_extraction(database)

    async def broken_send(*_args, **_kwargs):
        raise TimeoutError("verifier unreachable")

    result = verify(database, broken_send)

    assert result["status"] == "error"
    stored = database.request_by_id("r1")
    assert stored["verification_status"] == "error"
    assert "TimeoutError" in stored["verification_reason"]
    assert stored["output_text"] == SMALL_ANSWER  # the user's answer is untouched
