import pytest

from app.core.evaluation import (
    build_judge_prompt,
    evaluate_output,
    load_golden_dataset,
    parse_judge_result,
)


def test_exact_match_ignores_case_and_surrounding_whitespace():
    result = evaluate_output("42\n", [{"type": "exact_match", "value": "42"}])
    assert result["passed"] and result["score"] == 1.0


def test_json_values_accepts_fenced_json():
    check = {"type": "json_values", "values": {"name": "Asha Rao", "city": "Pune"}}
    assert evaluate_output('{"name":"Asha Rao","city":"Pune"}', [check])["passed"]
    assert evaluate_output('```json\n{"name":"Asha Rao","city":"Pune"}\n```', [check])["passed"]


def test_json_values_reports_mismatches_and_invalid_json():
    check = {"type": "json_values", "values": {"city": "Pune"}}
    mismatch = evaluate_output('{"city":"Delhi"}', [check])
    assert not mismatch["passed"] and "Mismatched" in mismatch["checks"][0]["detail"]
    invalid = evaluate_output("not json", [check])
    assert not invalid["passed"] and "Invalid JSON" in invalid["checks"][0]["detail"]


def test_contains_all_reports_what_is_missing():
    check = {"type": "contains_all", "values": ["library", "Tuesday"]}
    result = evaluate_output("library evening", [check])
    assert result["score"] == 0.0
    assert not result["passed"]
    assert "Tuesday" in result["checks"][0]["detail"]


def test_partial_score_is_the_fraction_of_checks_passed():
    checks = [
        {"type": "contains_all", "values": ["a"]},
        {"type": "contains_all", "values": ["zzz"]},
    ]
    assert evaluate_output("a", checks)["score"] == 0.5


def test_no_checks_never_passes():
    assert evaluate_output("anything", []) == {"passed": False, "score": 0.0, "checks": []}


def test_unknown_check_type_is_an_error():
    with pytest.raises(ValueError, match="Unsupported golden check type"):
        evaluate_output("x", [{"type": "vibes"}])


def test_every_bundled_golden_case_is_well_formed():
    cases = load_golden_dataset()
    assert cases
    assert len({case["case_id"] for case in cases}) == len(cases)
    for case in cases:
        # Raises if a case uses an unsupported check or is missing a required key.
        evaluate_output("", case["checks"])


def test_judge_prompt_marks_the_candidate_as_untrusted():
    prompt = build_judge_prompt("task", "answer", [{"type": "exact_match", "value": "1"}])
    assert "untrusted" in prompt and "task" in prompt and "answer" in prompt


def test_judge_score_is_validated():
    assert parse_judge_result('{"score":0.75,"reason":" Mostly correct. "}') == {
        "score": 0.75,
        "reason": "Mostly correct.",
    }
    for bad in ('{"reason":"no score"}', '{"score":"high"}', '{"score":1.5}'):
        with pytest.raises(ValueError):
            parse_judge_result(bad)
    with pytest.raises(TypeError):
        parse_judge_result('{"score":1,"reason":["not","a","string"]}')
