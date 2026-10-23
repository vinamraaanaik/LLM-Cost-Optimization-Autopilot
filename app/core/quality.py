"""Request-type inference and deterministic response-agreement scoring.

Agreement is a cheap, inspectable proxy for "did the small model match the stronger one".
It is not a semantic guarantee; thresholds live in data/quality_thresholds.json.
"""

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal, get_args

from app.core.config import DATA_DIR

RequestType = Literal[
    "auto",
    "extraction",
    "classification",
    "summarization",
    "structured_analysis",
    "creative_generation",
    "question_answering",
    "general",
]
REQUEST_TYPES = get_args(RequestType)
QUALITY_THRESHOLDS = json.loads((DATA_DIR / "quality_thresholds.json").read_text(encoding="utf-8"))

_WORD = re.compile(r"\w+", re.UNICODE)
_TYPE_PATTERNS = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("extraction", r"\b(?:extract|pull out|identify the fields|return the fields)\b"),
        ("classification", r"\b(?:classify|categorize|label this|sentiment)\b"),
        ("summarization", r"\b(?:summari[sz]e|summary|key points)\b"),
        ("structured_analysis", r"\b(?:analy[sz]e|compare|evaluate|assess|trade-offs)\b"),
        ("creative_generation", r"\b(?:write a story|brainstorm|creative|compose|fiction)\b"),
    )
)
_QUESTION = re.compile(r"\b(?:what|who|when|where|why|how|answer)\b")
_FIELD_PATTERNS = (
    re.compile(
        r"\b(?:extract|return|provide|include)\s+(?:only\s+)?(?:the\s+)?(.+?)\s+"
        r"\b(?:from|as|in|using)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\b(?:fields?|keys?)\s*(?:are|:)\s*(.+?)(?:[.!?\n]|$)", re.IGNORECASE | re.DOTALL),
)
_FORMAT_SUFFIX = re.compile(r"\b(?:as\s+)?(?:json|csv|xml|yaml)\b.*$", re.IGNORECASE)
_FIELD_SEPARATORS = re.compile(r",|\band\b|\bor\b|&", re.IGNORECASE)
_NOT_FIELDS = {"json", "csv", "xml", "yaml", "fields", "keys"}


@dataclass(frozen=True)
class QualityAssessment:
    request_type: str
    score: float
    threshold: float
    gap: float
    failed: bool
    reason: str


def infer_request_type(prompt: str, requested_type: str = "auto") -> str:
    if requested_type != "auto":
        if requested_type not in QUALITY_THRESHOLDS:
            raise ValueError(f"Unsupported request_type: {requested_type}")
        return requested_type
    lowered = prompt.casefold()
    for name, pattern in _TYPE_PATTERNS:
        if pattern.search(lowered):
            return name
    return "question_answering" if _QUESTION.search(lowered) else "general"


def _json_object(text: str) -> dict | None:
    """Parse a JSON object from a response, tolerating prose or code fences around it."""
    candidates = [text.strip()]
    if match := re.search(r"\{.*\}", text, re.DOTALL):
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_value(value) -> str:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True, ensure_ascii=False).casefold()
    return str(value).strip().casefold()


def _normalized_key(value: str) -> str:
    return " ".join(_WORD.findall(value.casefold()))


def _requested_fields(prompt: str) -> list[str]:
    """Best-effort parse of the field names a prompt asks to be extracted."""
    match = next((m for p in _FIELD_PATTERNS if (m := p.search(prompt))), None)
    if match is None:
        return []
    captured = _FORMAT_SUFFIX.sub("", match.group(1))
    fields = []
    for part in _FIELD_SEPARATORS.split(captured):
        field = re.sub(r"^(?:the|a|an)\s+", "", part.strip(), flags=re.IGNORECASE)
        field = re.sub(r"\s+", " ", re.sub(r"[^\w ]+", " ", field)).strip()
        if field and field.casefold() not in _NOT_FIELDS:
            fields.append(field)
    return fields


def _word_agreement(candidate: str, reference: str) -> float:
    """Mean of word-set F1 and in-order sequence similarity."""
    candidate_words = _WORD.findall(candidate.casefold())
    reference_words = _WORD.findall(reference.casefold())
    if not candidate_words and not reference_words:
        return 1.0
    if not candidate_words or not reference_words:
        return 0.0
    candidate_set, reference_set = set(candidate_words), set(reference_words)
    overlap = len(candidate_set & reference_set)
    precision = overlap / len(candidate_set)
    recall = overlap / len(reference_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sequence = SequenceMatcher(None, candidate_words, reference_words).ratio()
    return round((f1 + sequence) / 2, 4)


def _find_key(items: dict, wanted: str) -> str | None:
    """Match a normalized key exactly or as the last words of a longer key."""
    return next((key for key in items if key == wanted or key.endswith(f" {wanted}")), None)


def _extraction_score(prompt: str, candidate: str, reference: str) -> tuple[float, str]:
    candidate_json, reference_json = _json_object(candidate), _json_object(reference)
    if candidate_json is None or not reference_json:
        return (
            _word_agreement(candidate, reference),
            "JSON fields unavailable; used word overlap and sequence agreement",
        )
    candidate_items = {_normalized_key(str(k)): v for k, v in candidate_json.items()}
    reference_items = {_normalized_key(str(k)): v for k, v in reference_json.items()}
    requested = _requested_fields(prompt)
    expected = requested or list(reference_items)

    matched = 0
    for field in expected:
        wanted = _normalized_key(field)
        candidate_key = _find_key(candidate_items, wanted)
        if candidate_key is None:
            continue
        reference_key = _find_key(reference_items, wanted)
        if reference_key is None or _normalize_value(candidate_items[candidate_key]) == (
            _normalize_value(reference_items[reference_key])
        ):
            matched += 1
    source = "requested" if requested else "reference"
    score = matched / len(expected) if expected else 1.0
    return score, f"{matched}/{len(expected)} {source} fields and values matched"


def compare_outputs(
    prompt: str, candidate: str, reference: str, requested_type: str = "auto"
) -> QualityAssessment:
    request_type = infer_request_type(prompt, requested_type)
    threshold = float(QUALITY_THRESHOLDS[request_type]["minimum_agreement"])

    if request_type == "classification":
        score = float(_normalized_key(candidate) == _normalized_key(reference))
        detail = "normalized label match"
    elif request_type == "extraction":
        score, detail = _extraction_score(prompt, candidate, reference)
    else:
        score = _word_agreement(candidate, reference)
        detail = "word overlap and sequence agreement"

    score = round(float(score), 4)
    failed = score < threshold
    verdict = "below" if failed else "meets"
    return QualityAssessment(
        request_type=request_type,
        score=score,
        threshold=threshold,
        gap=round(max(0.0, threshold - score), 4),
        failed=failed,
        reason=f"{detail}; agreement {score:.0%} {verdict} the {threshold:.0%} threshold.",
    )
