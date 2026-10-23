import pytest

from app.core.quality import QUALITY_THRESHOLDS, REQUEST_TYPES, compare_outputs, infer_request_type


def test_every_request_type_has_a_threshold():
    assert set(REQUEST_TYPES) - {"auto"} == set(QUALITY_THRESHOLDS)


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Extract the date from this message.", "extraction"),
        ("Classify the sentiment of this review.", "classification"),
        ("Summarize this article.", "summarization"),
        ("Compare the trade-offs of A and B.", "structured_analysis"),
        ("Write a story about a robot.", "creative_generation"),
        ("Why is the sky blue?", "question_answering"),
        ("Hello there", "general"),
    ],
)
def test_request_type_is_inferred_from_the_prompt(prompt, expected):
    assert infer_request_type(prompt) == expected


def test_explicit_request_type_wins_and_is_validated():
    assert infer_request_type("Is the review positive?", "classification") == "classification"
    with pytest.raises(ValueError, match="Unsupported request_type"):
        infer_request_type("anything", "poetry")


def test_extraction_requires_every_field_and_value_to_match():
    result = compare_outputs(
        "Extract name and city as JSON.",
        '{"name":"Ada","city":"Paris"}',
        '{"name":"Ada","city":"London"}',
        "extraction",
    )
    assert result.failed
    assert result.score == 0.5
    assert result.gap > 0
    assert "1/2" in result.reason


def test_extraction_matches_keys_loosely_and_checks_requested_fields():
    result = compare_outputs(
        "Extract the name and date from this note. Return only JSON.",
        '{"Name": "Maya", "appointment date": "Nov 12"}',
        '{"name": "Maya", "date": "Nov 12", "extra": "ignored"}',
    )
    assert result.request_type == "extraction"
    assert not result.failed and "requested" in result.reason


def test_extraction_falls_back_to_word_agreement_without_json():
    result = compare_outputs(
        "Extract name and city as JSON.", "Ada lives in London", '{"name": "Ada"}', "extraction"
    )
    assert "JSON fields unavailable" in result.reason


def test_classification_normalizes_punctuation_but_catches_label_changes():
    same = compare_outputs("Classify the review.", "Positive.", "positive", "classification")
    different = compare_outputs("Classify the review.", "positive", "negative", "classification")
    assert not same.failed and same.score == 1.0
    assert different.failed and different.score == 0.0


def test_free_text_uses_overlap_against_the_type_threshold():
    same = compare_outputs("Summarize this.", "Evening hours rose", "Evening hours rose")
    different = compare_outputs("Summarize this.", "totally unrelated words", "Evening hours rose")
    assert same.score == 1.0 and not same.failed
    assert different.failed and different.gap > 0


def test_two_empty_answers_agree():
    assert compare_outputs("Why?", "", "").score == 1.0
