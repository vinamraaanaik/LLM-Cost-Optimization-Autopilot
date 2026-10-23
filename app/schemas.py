from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.core.config import DEFAULT_MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS, MAX_PROMPT_CHARS
from app.core.quality import RequestType

Prompt = Annotated[str, Field(min_length=1, max_length=MAX_PROMPT_CHARS)]
OutputTokens = Annotated[int, Field(ge=1, le=MAX_OUTPUT_TOKENS)]


class VerificationOptions(BaseModel):
    verify_quality: bool = True
    # How long to wait for verification before replying; 0 replies immediately and verifies
    # in the background.
    verification_budget_ms: int = Field(default=0, ge=0, le=30_000)


class GenerateRequest(VerificationOptions):
    prompt: Prompt
    model_id: str | None = None
    max_output_tokens: OutputTokens = DEFAULT_MAX_OUTPUT_TOKENS
    request_type: RequestType = "auto"


class GenerateResponse(BaseModel):
    request_id: str
    output_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    model_id: str
    complexity: str
    complexity_confidence: float
    route_reason: str
    used_fallback: bool
    original_model_id: str
    request_type: str
    verification_status: str
    verification_model_id: str | None = None
    quality_score: float | None = None
    quality_threshold: float | None = None
    quality_gap: float | None = None
    escalated: bool = False
    escalation_model_id: str | None = None
    escalation_cost_delta_usd: float = 0.0


class RouteRequest(BaseModel):
    prompt: Prompt


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=2_000)


class EvaluationRequest(BaseModel):
    model_id: str | None = None
    judge_model_id: str | None = None
    max_output_tokens: OutputTokens = DEFAULT_MAX_OUTPUT_TOKENS


class HealthResponse(BaseModel):
    status: str
    mode: str


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str = Field(max_length=MAX_PROMPT_CHARS)


class ChatCompletionRequest(VerificationOptions):
    # Accepted for OpenAI-client compatibility only; the router always picks the model.
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    max_tokens: int | None = Field(default=None, ge=1, le=MAX_OUTPUT_TOKENS)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=MAX_OUTPUT_TOKENS)
    stream: bool = False


class RoutingConfigUpdate(BaseModel):
    tier_to_models: dict[str, list[str]]
