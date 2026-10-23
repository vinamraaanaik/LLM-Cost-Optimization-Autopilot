import asyncio

import pytest
from fastapi import HTTPException

import app.main as main_module
from app.schemas import (
    ChatCompletionRequest,
    EvaluationRequest,
    FeedbackRequest,
    GenerateRequest,
    RouteRequest,
    RoutingConfigUpdate,
)
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


def status_of(call) -> int:
    with pytest.raises(HTTPException) as error:
        call()
    return error.value.status_code


def extraction_request(**overrides) -> GenerateRequest:
    return GenerateRequest(
        prompt=EXTRACTION_PROMPT, model_id="llama3.2:1b", request_type="extraction", **overrides
    )


# ---- generation and feedback -------------------------------------------------------------


def test_generate_records_the_request_and_accepts_feedback(monkeypatch, api_db):
    monkeypatch.setattr(main_module, "send_request", fake_send({"llama3.2:1b": "42"}))

    body = run(
        main_module.generate_completion(
            GenerateRequest(
                prompt="What is 6 multiplied by 7?", model_id="llama3.2:1b", verify_quality=False
            )
        )
    )

    assert (body.output_text, body.model_id) == ("42", "llama3.2:1b")
    assert body.verification_status == "disabled"
    assert body.total_tokens == body.input_tokens + body.output_tokens
    assert api_db.request_by_id(body.request_id)["prompt"] == "What is 6 multiplied by 7?"

    feedback = main_module.add_feedback(
        FeedbackRequest(request_id=body.request_id, rating=5, comment="great")
    )
    assert feedback["status"] == "saved"
    assert main_module.metrics()["request_count"] == 1
    assert main_module.metrics()["feedback_count"] == 1


def test_routed_generation_picks_a_local_model_and_explains_why(monkeypatch, api_db):
    monkeypatch.setattr(main_module, "send_request", fake_send())
    body = run(
        main_module.generate_completion(
            GenerateRequest(prompt="What is 2 plus 2?", verify_quality=False)
        )
    )
    assert body.model_id in {"llama3.2:1b", "llama3.1:8b"}
    assert body.original_model_id == body.model_id
    assert "lowest estimated-cost" in body.route_reason


def test_unknown_model_and_unknown_request_ids_are_404s(monkeypatch, api_db):
    monkeypatch.setattr(main_module, "send_request", fake_send())
    unknown_model = GenerateRequest(prompt="hi", model_id="nope")
    assert status_of(lambda: run(main_module.generate_completion(unknown_model))) == 404
    assert status_of(lambda: main_module.get_request("missing")) == 404
    missing_feedback = FeedbackRequest(request_id="missing", rating=3)
    assert status_of(lambda: main_module.add_feedback(missing_feedback)) == 404


def test_hosted_model_without_opt_in_is_a_503(api_db):
    request = GenerateRequest(prompt="hi", model_id="gpt-4o", verify_quality=False)
    assert status_of(lambda: run(main_module.generate_completion(request))) == 503


def test_unexpected_provider_failures_are_a_502_without_leaking_details(monkeypatch, api_db):
    async def broken(*_args, **_kwargs):
        raise ConnectionError("secret internal hostname")

    monkeypatch.setattr(main_module, "send_request", broken)
    request = GenerateRequest(prompt="hi", model_id="llama3.2:1b", verify_quality=False)
    with pytest.raises(HTTPException) as error:
        run(main_module.generate_completion(request))
    assert error.value.status_code == 502
    assert "ConnectionError" in error.value.detail
    assert "secret" not in error.value.detail


# ---- verification and escalation -----------------------------------------------------------


def test_response_returns_fast_and_the_background_verifier_escalates_later(monkeypatch, api_db):
    monkeypatch.setattr(main_module, "send_request", escalating_send())

    async def scenario():
        first = await main_module.generate_completion(extraction_request())
        # Verification is scheduled but has not finished: the caller still gets the first answer.
        assert first.verification_status == "pending"
        assert first.output_text == SMALL_ANSWER
        await asyncio.gather(*main_module._VERIFICATION_TASKS)
        return first

    first = run(scenario())

    final = main_module.get_request(first.request_id)
    assert final["verification_status"] == "escalated"
    assert final["output_text"] == FULL_ANSWER
    assert final["escalation_model_id"] == "llama3.1:8b"
    assert main_module._VERIFICATION_TASKS == set()  # finished tasks are released


def test_a_verification_budget_returns_the_escalated_answer_when_ready(monkeypatch, api_db):
    monkeypatch.setattr(main_module, "send_request", escalating_send())

    request = extraction_request(verification_budget_ms=2_000)
    response = run(main_module.generate_completion(request))

    assert response.verification_status == "escalated"
    assert response.model_id == "llama3.1:8b"
    assert response.original_model_id == "llama3.2:1b"
    assert response.output_text == FULL_ANSWER
    assert response.escalation_cost_delta_usd == 0.0004
    assert response.escalated and response.quality_gap > 0


def test_queue_mode_enqueues_a_job_and_replies_immediately(monkeypatch, api_db):
    monkeypatch.setenv("LLM_VERIFICATION_MODE", "queue")
    monkeypatch.setattr(main_module, "send_request", fake_send({"llama3.2:1b": "42"}))

    request = GenerateRequest(prompt="Reply with 42.", model_id="llama3.2:1b")
    body = run(main_module.generate_completion(request))

    assert body.verification_status == "pending"
    assert api_db.verification_job(body.request_id)["status"] == "queued"


def test_queue_mode_waits_for_the_worker_within_the_budget(monkeypatch, api_db):
    monkeypatch.setenv("LLM_VERIFICATION_MODE", "queue")
    monkeypatch.setattr(main_module, "send_request", fake_send())

    async def stand_in_worker():
        while not api_db.recent_requests():
            await asyncio.sleep(0.01)
        request_id = api_db.recent_requests()[0]["id"]
        await asyncio.sleep(0.1)
        api_db.update_verification(
            request_id, status="verified", request_type="general", reason="checked by the worker"
        )

    async def scenario():
        worker = asyncio.create_task(stand_in_worker())
        request = GenerateRequest(
            prompt="Reply with 42.", model_id="llama3.2:1b", verification_budget_ms=3_000
        )
        reply = await main_module.generate_completion(request)
        await worker
        return reply

    assert run(scenario()).verification_status == "verified"


def test_queue_mode_gives_up_waiting_when_the_budget_runs_out(monkeypatch, api_db):
    monkeypatch.setenv("LLM_VERIFICATION_MODE", "queue")
    monkeypatch.setattr(main_module, "send_request", fake_send())

    request = GenerateRequest(
        prompt="Reply with 42.", model_id="llama3.2:1b", verification_budget_ms=100
    )
    body = run(main_module.generate_completion(request))

    assert body.verification_status == "pending"  # no worker is running


# ---- OpenAI-style endpoint ---------------------------------------------------------------


def test_chat_completions_routes_and_returns_the_openai_shape(monkeypatch, api_db):
    monkeypatch.setenv("LLM_VERIFICATION_MODE", "queue")
    send = fake_send({"llama3.2:1b": "42", "llama3.1:8b": "42"})
    monkeypatch.setattr(main_module, "send_request", send)

    response = run(
        main_module.chat_completions(
            ChatCompletionRequest(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "Reply with the number."},
                ],
                max_tokens=24,
            )
        )
    )

    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"] == {"role": "assistant", "content": "42"}
    assert response["model"] != "gpt-4o"  # the client's model hint is ignored by design
    assert response["metadata"]["selected_model"] == response["model"]
    assert response["metadata"]["route_reason"]
    assert response["usage"] == {"prompt_tokens": 8, "completion_tokens": 7, "total_tokens": 15}
    assert send.calls[0][0] == "SYSTEM: Be terse.\n\nUSER: Reply with the number."
    assert api_db.verification_job(response["metadata"]["request_id"])["status"] == "queued"


def test_chat_completions_rejects_unsupported_requests():
    streaming = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], stream=True)
    assert status_of(lambda: run(main_module.chat_completions(streaming))) == 400
    no_user = ChatCompletionRequest(messages=[{"role": "assistant", "content": "hi"}])
    assert status_of(lambda: run(main_module.chat_completions(no_user))) == 422


# ---- read-only endpoints -----------------------------------------------------------------


def test_route_endpoint_is_local_and_never_calls_a_provider():
    response = main_module.route_request(RouteRequest(prompt="What is 2 plus 2?"))
    assert response["model"]["local"] is True
    assert response["estimated_cost_usd"] == 0.0


def test_models_endpoint_reports_availability_under_current_settings(monkeypatch):
    models = {m["model_id"]: m for m in main_module.list_models()}
    assert models["llama3.2:1b"]["available_for_routing"]
    assert not models["gpt-4o"]["available_for_routing"]
    assert models["gpt-4o"]["cost_per_million_output_tokens_usd"] == 10.0
    monkeypatch.setenv("LLM_ALLOW_PAID_PROVIDERS", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert {m["model_id"]: m for m in main_module.list_models()}["gpt-4o"]["available_for_routing"]


def test_health_and_history_endpoints(api_db):
    assert main_module.health().status == "ok"
    api_db.record_request(request_row())
    assert [row["id"] for row in main_module.recent_requests(limit=10)] == ["r1"]
    assert main_module.recent_routing_failures(limit=10) == []


def test_stats_never_fabricate_savings_without_requests(api_db):
    empty = main_module.cost_savings_stats()
    assert empty["status"] == "no_inference_data"
    assert empty["estimated_savings_usd"] is None
    assert empty["estimated_cost_reduction_pct"] is None

    api_db.record_request(request_row(id="measured", input_tokens=1_000, output_tokens=1_000))
    measured = main_module.cost_savings_stats()
    assert measured["status"] == "estimate"
    assert measured["hypothetical_baseline_cost_usd"] == 0.0125  # 1k in @ $2.50/M + 1k out @ $10/M
    assert measured["estimated_savings_usd"] == 0.0125
    assert measured["estimated_cost_reduction_pct"] == 100.0


# ---- routing config ----------------------------------------------------------------------


def test_routing_config_endpoint_persists_the_live_mapping(monkeypatch, tmp_path):
    rules_path = tmp_path / "runtime-rules.yaml"
    monkeypatch.setenv("LLM_ROUTING_RULES_PATH", str(rules_path))
    mapping = {
        "simple": ["llama3.2:1b"],
        "moderate": ["llama3.1:8b"],
        "complex": ["claude-sonnet-5"],
    }

    response = main_module.put_routing_config(RoutingConfigUpdate(tier_to_models=mapping))

    assert response["tier_to_models"] == mapping
    assert main_module.get_routing_config()["tier_to_models"] == mapping
    assert rules_path.exists()


def test_invalid_routing_config_is_a_422(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_ROUTING_RULES_PATH", str(tmp_path / "rules.yaml"))
    bad = RoutingConfigUpdate(
        tier_to_models={"simple": ["no-such-model"], "moderate": ["x"], "complex": ["y"]}
    )
    assert status_of(lambda: main_module.put_routing_config(bad)) == 422


# ---- golden-set evaluation ---------------------------------------------------------------


GOLDEN_ANSWERS = (
    ("6 multiplied by 7", "42"),
    ("Classify this message", "positive"),
    ("Return only JSON", '{"name":"Asha Rao","city":"Pune"}'),
    ("Summarize in one sentence", "The library added evening hours, increasing visits."),
    ("Python function", "def is_even(value):\n    return value % 2 == 0"),
)
PERFECT_JUDGE_VERDICT = '{"score":1,"reason":"All required checks are met."}'


def golden_send():
    """Answers every bundled golden case correctly, and plays a judge that approves."""

    async def send(prompt, model_config, **_):
        if prompt.startswith("Evaluate the candidate response"):
            return response(model_config.model_id, PERFECT_JUDGE_VERDICT)
        answer = next(text for marker, text in GOLDEN_ANSWERS if marker in prompt)
        return response(model_config.model_id, answer)

    return send


def test_golden_evaluation_supports_an_optional_local_judge(monkeypatch, api_db):
    monkeypatch.setattr(main_module, "send_request", golden_send())

    result = run(
        main_module.run_evaluation(
            EvaluationRequest(model_id="llama3.2:1b", judge_model_id="llama3.2:1b")
        )
    )

    assert result["average_score"] == 1.0
    assert result["passed_count"] == result["case_count"] == 5
    assert result["average_judge_score"] == 1.0
    assert main_module.metrics()["judge_evaluation_count"] == 5


def test_evaluation_without_a_judge_reports_no_judge_score(monkeypatch, api_db):
    monkeypatch.setattr(main_module, "send_request", golden_send())
    result = run(main_module.run_evaluation(EvaluationRequest()))
    assert result["model_id"] == "routed-local"
    assert result["average_judge_score"] is None


def test_unparseable_judge_output_is_recorded_not_fatal(monkeypatch, api_db):
    monkeypatch.setattr(main_module, "send_request", fake_send({"llama3.2:1b": "not json"}))
    result = run(
        main_module.run_evaluation(
            EvaluationRequest(model_id="llama3.2:1b", judge_model_id="llama3.2:1b")
        )
    )
    assert result["cases"][0]["judge_score"] is None
    assert "could not be parsed" in result["cases"][0]["judge_reason"]


def test_evaluation_validates_models_up_front(api_db):
    unknown = EvaluationRequest(model_id="nope")
    assert status_of(lambda: run(main_module.run_evaluation(unknown))) == 404
    unknown_judge = EvaluationRequest(judge_model_id="nope")
    assert status_of(lambda: run(main_module.run_evaluation(unknown_judge))) == 404
    hosted_judge = EvaluationRequest(judge_model_id="gpt-4o")
    assert status_of(lambda: run(main_module.run_evaluation(hosted_judge))) == 400
