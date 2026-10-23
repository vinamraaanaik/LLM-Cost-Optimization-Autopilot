"""Prompt-complexity classifier: hand-crafted features + logistic regression.

Training data is the hand-labeled TSV plus any routing failures recorded in the database.
Features are always computed from the prompt text, so training and serving cannot drift apart.
"""

import csv
import re
import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import DATA_DIR

EXAMPLES_PATH = DATA_DIR / "complexity_examples.tsv"
LABEL_ORDER = ("simple", "moderate", "complex")
LABEL_SCORE = {"simple": 0.15, "moderate": 0.5, "complex": 0.85}
FEATURE_NAMES = (
    "token_count",
    "instruction_terms",
    "instruction_count",
    "constraint_count",
    "context_provided",
    "output_format_complexity",
)
MIN_EXAMPLES_PER_LABEL = 60

INSTRUCTION_TERMS = (
    "analyze",
    "assess",
    "classify",
    "compare",
    "create",
    "critique",
    "derive",
    "design",
    "evaluate",
    "explain",
    "extract",
    "format",
    "generate",
    "judge",
    "plan",
    "prioritize",
    "rank",
    "recommend",
    "reason",
    "rewrite",
    "summarize",
    "translate",
)
_INSTRUCTION_PATTERNS = {term: re.compile(rf"\b{re.escape(term)}\b") for term in INSTRUCTION_TERMS}
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CONSTRAINT = re.compile(
    r"\b(?:must|include|exclude|avoid|preserve|exactly|at least|no more than|within"
    r"|return only|keep|do not|don't)\b",
    re.IGNORECASE,
)
_CONJUNCTION = re.compile(r"\b(?:and|while|whereas|but also)\b|;")
_CONTEXT = re.compile(
    r"\b(?:context|text|note|passage|email|review|article|transcript|following|below|"
    r"provided|sentence|paragraph|message|code|function|source|report|table|quote|data)\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"['\"“”].+['\"“”]")
_FORMATS = {
    "json": re.compile(r"\bjson\b", re.IGNORECASE),
    "xml": re.compile(r"\bxml\b", re.IGNORECASE),
    "csv": re.compile(r"\bcsv\b", re.IGNORECASE),
    "yaml": re.compile(r"\byaml\b", re.IGNORECASE),
    "code": re.compile(r"\b(?:code|function|script|program)\b|```", re.IGNORECASE),
    "table": re.compile(r"\btable\b|tabular", re.IGNORECASE),
    "list": re.compile(r"\b(?:bullets?|numbered list|steps|checklist)\b", re.IGNORECASE),
    "schema": re.compile(r"\bschema\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class ComplexityResult:
    label: str
    confidence: float
    score: float
    signals: tuple[str, ...] = ()


def _format_complexity(prompt: str) -> int:
    """0 = no requested format, 1 = a loose list, 2 = one structured format, 3 = several."""
    matched = [name for name, pattern in _FORMATS.items() if pattern.search(prompt)]
    if len(matched) >= 2:
        return 3
    if matched == ["list"]:
        return 1
    return 2 if matched else 0


def extract_features(prompt: str) -> dict:
    """Auditable, deterministic features; the same function feeds training and serving."""
    lowered = prompt.casefold()
    terms = [term for term, pattern in _INSTRUCTION_PATTERNS.items() if pattern.search(lowered)]
    constraints = len(_CONSTRAINT.findall(prompt)) + len(_CONJUNCTION.findall(lowered))
    return {
        "token_count": len(_TOKEN.findall(prompt)),
        "instruction_terms": terms,
        "instruction_count": len(terms),
        "constraint_count": min(constraints, 12),
        "context_provided": int(bool(_CONTEXT.search(prompt) or _QUOTED.search(prompt))),
        "output_format_complexity": _format_complexity(prompt),
    }


def _model_features(features: dict) -> dict:
    """Flatten the instruction-term list into one-hot keys for DictVectorizer."""
    values = {key: value for key, value in features.items() if key != "instruction_terms"}
    values.update({f"instruction:{term}": 1 for term in features["instruction_terms"]})
    return values


def load_labeled_examples(path: Path = EXAMPLES_PATH) -> list[dict]:
    """Read the hand-labeled prompts and enforce basic dataset-quality gates."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    counts = Counter(row["label"] for row in rows)
    if set(counts) - set(LABEL_ORDER):
        raise ValueError(f"Dataset contains unsupported tier labels: {sorted(counts)}")
    if any(counts[label] < MIN_EXAMPLES_PER_LABEL for label in LABEL_ORDER):
        raise ValueError(
            f"Need at least {MIN_EXAMPLES_PER_LABEL} labeled prompts per tier; got {dict(counts)}"
        )
    return [{"text": r["prompt"], "label": r["label"], "source": "hand_labeled"} for r in rows]


def training_samples(failures: Iterable[dict] = ()) -> list[dict]:
    """Hand-labeled examples plus routing failures, relabeled with the tier that was needed."""
    samples = load_labeled_examples()
    samples += [
        {"text": failure["prompt"], "label": failure["label"], "source": "routing_failure"}
        for failure in failures
        if failure["label"] in LABEL_ORDER
    ]
    return samples


def _new_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("features", DictVectorizer()),
            ("scale", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42)),
        ]
    )


def _matrix(samples: list[dict]) -> tuple[list[dict], list[str]]:
    features = [_model_features(extract_features(sample["text"])) for sample in samples]
    return features, [sample["label"] for sample in samples]


def evaluate_classifier(
    samples: list[dict] | None = None, *, test_size: float = 0.2, random_state: int = 42
) -> dict:
    """Reproducible stratified hold-out report with a tier-ordered confusion matrix."""
    samples = training_samples() if samples is None else samples
    features, labels = _matrix(samples)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    predictions = _new_pipeline().fit(train_x, train_y).predict(test_x)
    return {
        "dataset_size": len(samples),
        "train_size": len(train_y),
        "test_size": len(test_y),
        "accuracy": round(float(accuracy_score(test_y, predictions)), 4),
        "label_order": list(LABEL_ORDER),
        "confusion_matrix": confusion_matrix(test_y, predictions, labels=LABEL_ORDER).tolist(),
        "classification_report": classification_report(
            test_y, predictions, labels=LABEL_ORDER, output_dict=True, zero_division=0
        ),
        "label_counts": dict(Counter(labels)),
        "source_counts": dict(Counter(sample["source"] for sample in samples)),
        "feature_names": list(FEATURE_NAMES),
    }


# The serving model is replaced wholesale on retrain, so in-flight requests never observe a
# half-trained pipeline.
_serving_model: Pipeline | None = None
_serving_lock = threading.Lock()


def _publish(model: Pipeline) -> Pipeline:
    global _serving_model
    _serving_model = model
    return model


def load_serving_classifier(failures: Iterable[dict] = ()) -> Pipeline:
    """Fit the serving classifier on all available training data and publish it."""
    features, labels = _matrix(training_samples(failures))
    return _publish(_new_pipeline().fit(features, labels))


def retrain(failures: Iterable[dict] = ()) -> dict:
    """Refit the serving classifier and return the hold-out report for the same data."""
    failures = list(failures)
    metrics = evaluate_classifier(training_samples(failures))
    load_serving_classifier(failures)
    return metrics


def _classifier() -> Pipeline:
    if _serving_model is None:
        with _serving_lock:
            if _serving_model is None:
                load_serving_classifier()
    return _serving_model


def classify_prompt(prompt: str) -> ComplexityResult:
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    features = extract_features(prompt)
    classifier = _classifier()
    probabilities = classifier.predict_proba([_model_features(features)])[0]
    classes = [str(name) for name in classifier.named_steps["model"].classes_]
    best = int(probabilities.argmax())
    weighted = zip(probabilities, classes, strict=True)
    score = sum(float(probability) * LABEL_SCORE[name] for probability, name in weighted)

    signals = [f"classifier predicted {classes[best]}"]
    if features["instruction_terms"]:
        signals.append("instructions: " + ", ".join(features["instruction_terms"][:5]))
    if features["constraint_count"]:
        signals.append(f"{features['constraint_count']} estimated constraints")
    if features["context_provided"]:
        signals.append("provided context detected")
    return ComplexityResult(
        label=classes[best],
        confidence=round(float(probabilities[best]), 4),
        score=round(score, 4),
        signals=tuple(signals),
    )
