"""FastAPI service: cost-aware routing, quality verification, and evaluation."""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager, contextmanager, suppress
from datetime import UTC, datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from app import __version__
from app.core.catalog import MODEL_REGISTRY, ModelConfig
from app.core.complexity import ComplexityResult, classify_prompt, load_serving_classifier, retrain
from app.core.config import DEFAULT_MAX_OUTPUT_TOKENS, MAX_PROMPT_CHARS
from app.core.costing import estimate_cost_usd, savings_vs_baseline
from app.core.database import PENDING_VERIFICATION, Database
from app.core.drift import detect_drift
from app.core.evaluation import (
    build_judge_prompt,
    evaluate_output,
    load_golden_dataset,
    parse_judge_result,
)
from app.core.quality import infer_request_type
from app.core.router import choose_model, load_routing_rules, update_tier_to_models
from app.core.verification import verify_and_escalate
from app.providers.base import ProviderConfigurationError, Response
from app.providers.unified import send_request
from app.schemas import (
    ChatCompletionRequest,
    EvaluationRequest,
    FeedbackRequest,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    RouteRequest,
    RoutingConfigUpdate,
)

logger = logging.getLogger("llm_cost_autopilot.api")

DEFAULT_RETRAIN_INTERVAL_SECONDS = 7 * 24 * 60 * 60
BASELINE_MODEL_ID = "gpt-4o"  # what /v1/stats prices all traffic at for the savings estimate
JUDGE_MAX_OUTPUT_TOKENS = 256
HistoryLimit = Annotated[int, Query(ge=1, le=500)]

database = Database()
_VERIFICATION_TASKS: set[asyncio.Task] = set()


# ---- lifecycle -------------------------------------------------------------------------


def _retrain_now() -> dict:
    return retrain(database.training_failures())


async def _retrain_periodically() -> None:
    """Refit the classifier on a schedule so recorded routing failures improve routing."""
    interval = float(
        os.getenv("LLM_RETRAIN_INTERVAL_SECONDS", str(DEFAULT_RETRAIN_INTERVAL_SECONDS))
    )
    while True:
        await asyncio.sleep(interval)
        try:
            metrics = await asyncio.to_thread(_retrain_now)
            logger.info(
                "Classifier retrained on %d examples; hold-out accuracy %.1f%%",
                metrics["dataset_size"],
                metrics["accuracy"] * 100,
            )
        except Exception:
            logger.exception("Classifier retraining failed")


@asynccontextmanager
async def lifespan(_app):
    # Fit before serving so the first request doesn't pay for training.
    await asyncio.to_thread(lambda: load_serving_classifier(database.training_failures()))
    retraining = asyncio.create_task(_retrain_periodically())
    try:
        yield
    finally:
        retraining.cancel()
        with suppress(asyncio.CancelledError):
            await retraining


app = FastAPI(
    title="LLM Cost Autopilot",
    description="Local-first routing, cost tracking, and quality evaluation for LLMs.",
    version=__version__,
    lifespan=lifespan,
)


# ---- helpers ---------------------------------------------------------------------------


def _availability_note(model: ModelConfig) -> str:
    if model.local:
        return "Configured for local Ollama; pull the model before inference."
    if model.is_available():
        return "Hosted provider is enabled and credentials are configured."
    return "Set the paid-provider opt-in and provider API key to make this eligible."


def _model_summary(model: ModelConfig) -> dict:
    return {
        "provider_name": model.provider_name,
        "model_id": model.model_id,
        "quality_tier": model.quality_tier.value,
        "local": model.local,
        "cost_per_input_token_usd": model.cost_per_input_token,
        "cost_per_output_token_usd": model.cost_per_output_token,
        "cost_per_million_input_tokens_usd": round(model.cost_per_input_token * 1_000_000, 6),
        "cost_per_million_output_tokens_usd": round(model.cost_per_output_token * 1_000_000, 6),
        "available_for_routing": model.is_available(),
        "availability_note": _availability_note(model),
    }


def _require_model(model_id: str, field: str = "model_id") -> ModelConfig:
    model = MODEL_REGISTRY.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Unknown {field}.")
    return model


@contextmanager
def _provider_errors(failure_label: str):
    """Translate configuration and provider failures into the HTTP errors clients should see."""
    try:
        yield
    except HTTPException:
        raise
    except (ProviderConfigurationError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        detail = f"{failure_label} ({type(error).__name__})."
        raise HTTPException(status_code=502, detail=detail) from error


def _select_model(
    request: GenerateRequest, complexity: ComplexityResult
) -> tuple[ModelConfig, str, bool]:
    """Return (model, route reason, used_fallback) for an explicit or routed request."""
    if request.model_id:
        model = _require_model(request.model_id)
        return model, f"Explicit model selection: {request.model_id}.", False
    decision = choose_model(complexity, prompt=request.prompt)
    return decision.model, decision.reason, decision.used_fallback


def _verification_task_done(task: asyncio.Task) -> None:
    _VERIFICATION_TASKS.discard(task)
    try:
        task.result()
    except Exception:
        logger.exception("Unhandled verification job failure")


async def _wait_for_queued_verification(request_id: str, budget_seconds: float) -> None:
    """Poll the database until the worker settles the request or the budget runs out."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget_seconds
    while (remaining := deadline - loop.time()) > 0:
        await asyncio.sleep(min(0.05, remaining))
        row = database.request_by_id(request_id)
        if row and row["verification_status"] not in PENDING_VERIFICATION:
            return


async def _start_verification(
    request: GenerateRequest,
    request_id: str,
    request_type: str,
    complexity: str,
    model: ModelConfig,
    result: Response,
) -> None:
    """Verify in the worker queue or in-process, optionally waiting up to the request budget."""
    budget_seconds = request.verification_budget_ms / 1_000
    if os.getenv("LLM_VERIFICATION_MODE", "inline").strip().lower() == "queue":
        database.enqueue_verification_job(request_id, request.max_output_tokens)
        if budget_seconds:
            await _wait_for_queued_verification(request_id, budget_seconds)
        return

    task = asyncio.create_task(
        verify_and_escalate(
            database=database,
            request_id=request_id,
            prompt=request.prompt,
            request_type=request_type,
            complexity=complexity,
            original_model=model,
            original_response=result,
            max_output_tokens=request.max_output_tokens,
            send=send_request,
        )
    )
    _VERIFICATION_TASKS.add(task)  # keep a reference so the task isn't garbage-collected
    task.add_done_callback(_verification_task_done)
    if budget_seconds:
        await asyncio.wait({task}, timeout=budget_seconds)


# ---- routes ----------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", mode="local-first")


@app.get("/v1/models")
def list_models() -> list[dict]:
    return [_model_summary(model) for model in MODEL_REGISTRY.values()]


@app.get("/v1/routing-config")
def get_routing_config() -> dict:
    return {"tier_to_models": load_routing_rules()["model_candidates_by_tier"]}


@app.put("/v1/routing-config")
def put_routing_config(request: RoutingConfigUpdate) -> dict:
    try:
        rules = update_tier_to_models(request.tier_to_models)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"tier_to_models": rules["model_candidates_by_tier"]}


@app.post("/v1/route")
def route_request(request: RouteRequest) -> dict:
    """Show the routing decision for a prompt without calling any model."""
    try:
        complexity = classify_prompt(request.prompt)
        decision = choose_model(complexity, prompt=request.prompt)
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "complexity": complexity.label,
        "complexity_confidence": complexity.confidence,
        "complexity_score": complexity.score,
        "model": _model_summary(decision.model),
        "estimated_cost_usd": decision.estimated_cost_usd,
        "requested_tier": decision.requested_tier.value,
        "selected_tier": decision.selected_tier.value,
        "route_reason": decision.reason,
        "used_fallback": decision.used_fallback,
    }


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate_completion(request: GenerateRequest) -> GenerateResponse:
    with _provider_errors("Provider request failed"):
        complexity = classify_prompt(request.prompt)
        request_type = infer_request_type(request.prompt, request.request_type)
        model, route_reason, used_fallback = _select_model(request, complexity)
        result = await send_request(
            request.prompt, model, max_output_tokens=request.max_output_tokens
        )

    request_id = str(uuid.uuid4())
    database.record_request(
        {
            "id": request_id,
            "created_at": datetime.now(UTC).isoformat(),
            "prompt": request.prompt,
            "model_id": result.model_id,
            "provider_name": model.provider_name,
            "complexity": complexity.label,
            "confidence": complexity.confidence,
            "route_reason": route_reason,
            "used_fallback": int(used_fallback),
            "output_text": result.output_text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd,
            "original_model_id": model.model_id,
            "request_type": request_type,
            "verification_status": "pending" if request.verify_quality else "disabled",
        }
    )
    if request.verify_quality:
        await _start_verification(
            request, request_id, request_type, complexity.label, model, result
        )

    # Re-read the row: verification may already have replaced the answer with an escalated one.
    row = database.request_by_id(request_id)
    return GenerateResponse(
        request_id=request_id,
        output_text=row["output_text"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["input_tokens"] + row["output_tokens"],
        latency_ms=row["latency_ms"],
        cost_usd=row["cost_usd"],
        model_id=row["model_id"],
        complexity=complexity.label,
        complexity_confidence=complexity.confidence,
        route_reason=route_reason,
        used_fallback=used_fallback,
        original_model_id=model.model_id,
        request_type=row["request_type"],
        verification_status=row["verification_status"],
        verification_model_id=row["verification_model_id"],
        quality_score=row["quality_score"],
        quality_threshold=row["quality_threshold"],
        quality_gap=row["quality_gap"],
        escalated=bool(row["escalated"]),
        escalation_model_id=row["escalation_model_id"],
        escalation_cost_delta_usd=row["escalation_cost_delta_usd"],
    )


@app.post("/v1/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict:
    """OpenAI-style text chat endpoint; the router always chooses the model."""
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported in this version.")
    if not any(message.role == "user" for message in request.messages):
        raise HTTPException(status_code=422, detail="At least one user message is required.")
    prompt = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in request.messages)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Combined message content exceeds {MAX_PROMPT_CHARS:,} characters.",
        )
    result = await generate_completion(
        GenerateRequest(
            prompt=prompt,
            max_output_tokens=(
                request.max_completion_tokens or request.max_tokens or DEFAULT_MAX_OUTPUT_TOKENS
            ),
            verify_quality=request.verify_quality,
            verification_budget_ms=request.verification_budget_ms,
        )
    )
    return {
        "id": f"chatcmpl-{result.request_id}",
        "object": "chat.completion",
        "created": int(datetime.now(UTC).timestamp()),
        "model": result.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
        },
        "metadata": {
            "request_id": result.request_id,
            "selected_model": result.model_id,
            "original_model_id": result.original_model_id,
            "complexity": result.complexity,
            "complexity_confidence": result.complexity_confidence,
            "route_reason": result.route_reason,
            "cost_usd_estimate": result.cost_usd,
            "latency_ms": result.latency_ms,
            "verification_status": result.verification_status,
            "quality_score": result.quality_score,
            "escalated": result.escalated,
        },
    }


@app.post("/v1/feedback")
def add_feedback(request: FeedbackRequest) -> dict:
    """Human-in-the-loop signal: store a 1-5 rating for a generated response."""
    if not database.add_feedback(request.request_id, request.rating, request.comment):
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    return {"status": "saved", "request_id": request.request_id}


@app.get("/v1/requests")
def recent_requests(limit: HistoryLimit = 50) -> list[dict]:
    return database.recent_requests(limit)


@app.get("/v1/requests/{request_id}")
def get_request(request_id: str) -> dict:
    row = database.request_by_id(request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    return row


@app.get("/v1/failures")
def recent_routing_failures(limit: HistoryLimit = 50) -> list[dict]:
    return database.recent_routing_failures(limit)


@app.get("/v1/metrics")
def metrics() -> dict:
    return {**database.metrics(), "drift": detect_drift(database.evaluation_scores())}


@app.get("/v1/stats")
def cost_savings_stats() -> dict:
    """Estimate routed spend against an all-GPT-4o baseline from stored token counts only."""
    baseline = MODEL_REGISTRY.get(BASELINE_MODEL_ID)
    if baseline is None:
        raise HTTPException(status_code=503, detail="GPT-4o baseline pricing is not configured.")
    usage = database.cost_usage_summary()
    has_data = usage["request_count"] > 0
    baseline_cost = estimate_cost_usd(baseline, usage["input_tokens"], usage["output_tokens"])
    savings, reduction_pct = savings_vs_baseline(usage["actual_cost_usd"], baseline_cost)
    return {
        "status": "estimate" if has_data else "no_inference_data",
        "request_count": usage["request_count"],
        "baseline_model_id": baseline.model_id,
        "actual_estimated_cost_usd": usage["actual_cost_usd"] if has_data else None,
        "hypothetical_baseline_cost_usd": baseline_cost if has_data else None,
        "estimated_savings_usd": round(savings, 10) if has_data else None,
        "estimated_cost_reduction_pct": (
            round(reduction_pct, 2) if reduction_pct is not None else None
        ),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "note": (
            "Hypothetical estimate: recorded request token counts are priced at current "
            "GPT-4o rates; verifier tokens are included. No model call is made."
            if has_data
            else "No inference requests are recorded; savings are unavailable, not zero."
        ),
    }


def _judge_verdict(response: Response) -> dict:
    try:
        return {**parse_judge_result(response.output_text), "model_id": response.model_id}
    except (ValueError, TypeError) as error:
        reason = f"Judge output could not be parsed: {error}"
        return {"score": None, "reason": reason, "model_id": response.model_id}


async def _evaluate_case(
    case: dict, model: ModelConfig | None, judge_model: ModelConfig | None, max_output_tokens: int
) -> dict:
    """Run one golden case: generate, apply deterministic checks, optionally ask a judge."""
    if model is None:  # route each golden prompt locally; evaluation never spends money
        decision = choose_model(
            classify_prompt(case["prompt"]), allow_paid=False, prompt=case["prompt"]
        )
        model = decision.model
    response = await send_request(case["prompt"], model, max_output_tokens=max_output_tokens)
    result = evaluate_output(response.output_text, case["checks"])

    judge, judge_cost_usd = None, 0.0
    if judge_model is not None:
        judge_response = await send_request(
            build_judge_prompt(case["prompt"], response.output_text, case["checks"]),
            judge_model,
            max_output_tokens=min(max_output_tokens, JUDGE_MAX_OUTPUT_TOKENS),
        )
        judge, judge_cost_usd = _judge_verdict(judge_response), judge_response.cost_usd
    return {
        "case_id": case["case_id"],
        "model_id": response.model_id,
        "output_text": response.output_text,
        **result,
        "judge": judge,
        "judge_score": judge["score"] if judge else None,
        "judge_model_id": judge["model_id"] if judge else None,
        "judge_reason": judge["reason"] if judge else None,
        "model_cost_usd": response.cost_usd,
        "judge_cost_usd": judge_cost_usd,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_ms": response.latency_ms,
    }


@app.post("/v1/evaluate")
async def run_evaluation(request: EvaluationRequest) -> dict:
    """Run the golden set through a model (or the local router) with deterministic checks."""
    model = _require_model(request.model_id) if request.model_id else None
    judge_model = None
    if request.judge_model_id:
        judge_model = _require_model(request.judge_model_id, "judge_model_id")
        if not judge_model.local:
            raise HTTPException(status_code=400, detail="The optional judge must be a local model.")

    rows = []
    with _provider_errors("Evaluation call failed"):
        for case in load_golden_dataset():
            rows.append(await _evaluate_case(case, model, judge_model, request.max_output_tokens))

    run_id = str(uuid.uuid4())
    database.record_evaluation_run(run_id, rows)
    judge_scores = [row["judge_score"] for row in rows if row["judge_score"] is not None]
    return {
        "run_id": run_id,
        "model_id": request.model_id or "routed-local",
        "case_count": len(rows),
        "average_score": round(sum(row["score"] for row in rows) / len(rows), 4) if rows else 0.0,
        "passed_count": sum(row["passed"] for row in rows),
        "average_judge_score": (
            round(sum(judge_scores) / max(1, len(judge_scores)), 4)
            if request.judge_model_id
            else None
        ),
        "cases": rows,
    }
