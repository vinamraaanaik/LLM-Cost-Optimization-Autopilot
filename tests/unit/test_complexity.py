import pytest

from app.core import complexity
from app.core.complexity import (
    FEATURE_NAMES,
    LABEL_ORDER,
    classify_prompt,
    evaluate_classifier,
    extract_features,
    load_labeled_examples,
    retrain,
    training_samples,
)


def test_classifier_returns_supported_label_and_confidence():
    result = classify_prompt("What is the capital of France?")
    assert result.label in LABEL_ORDER
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.score <= 1.0


def test_classifier_rejects_blank_prompts():
    with pytest.raises(ValueError, match="empty"):
        classify_prompt("   ")


def test_feature_extraction_is_stable():
    # Features are computed at both training and serving time; pin them so a change to the
    # extractor is a deliberate decision (and a prompt to refresh the metrics report).
    assert extract_features(
        "Extract the name and date from this note as JSON; do not include extra text."
    ) == {
        "token_count": 17,
        "instruction_terms": ["extract"],
        "instruction_count": 1,
        "constraint_count": 4,
        "context_provided": 1,
        "output_format_complexity": 2,
    }
    assert extract_features("Summarize the following article in three bullets.") == {
        "token_count": 8,
        "instruction_terms": ["summarize"],
        "instruction_count": 1,
        "constraint_count": 0,
        "context_provided": 1,
        "output_format_complexity": 1,
    }
    assert set(extract_features("anything")) == set(FEATURE_NAMES)


def test_holdout_metrics_include_accuracy_and_confusion_matrix():
    result = evaluate_classifier()
    assert result["dataset_size"] >= 200
    assert result["test_size"] > 0
    assert result["accuracy"] >= 0.8
    assert len(result["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in result["confusion_matrix"])
    assert result["source_counts"] == {"hand_labeled": result["dataset_size"]}


def test_labeled_dataset_covers_every_tier():
    samples = load_labeled_examples()
    assert len(samples) >= 200
    assert {sample["label"] for sample in samples} == set(LABEL_ORDER)


def test_routing_failures_extend_the_training_data():
    failures = [
        {"prompt": "A prompt the small model got wrong", "label": "moderate"},
        {"prompt": "Ignored: unsupported label", "label": "unknown"},
    ]
    samples = training_samples(failures)
    assert len(samples) == len(training_samples()) + 1
    assert samples[-1] == {
        "text": "A prompt the small model got wrong",
        "label": "moderate",
        "source": "routing_failure",
    }


def test_retrain_reports_sources_and_replaces_the_serving_model(monkeypatch):
    monkeypatch.setattr(complexity, "_serving_model", None)
    metrics = retrain([{"prompt": "Escalated prompt", "label": "complex"}])
    assert metrics["source_counts"]["routing_failure"] == 1
    assert complexity._serving_model is not None


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["label\tprompt", "simple\ta", "bogus\tb"], "unsupported"),
        (["label\tprompt", "simple\ta", "moderate\tb", "complex\tc"], "at least"),
    ],
)
def test_dataset_quality_gates(tmp_path, rows, message):
    path = tmp_path / "examples.tsv"
    path.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_labeled_examples(path)
