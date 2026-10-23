"""The load test's data generation and reporting are pure functions; the HTTP run is not tested."""

from collections import Counter
from datetime import UTC, datetime

from benchmarks.run_load_test import (
    RunSettings,
    build_prompt_set,
    percentile,
    render_report,
    summarize,
)

SEED = 20261021


def test_prompts_are_deterministic_per_seed():
    assert build_prompt_set(36, SEED) == build_prompt_set(36, SEED)
    assert build_prompt_set(36, SEED) != build_prompt_set(36, SEED + 1)


def test_the_seed_contract_is_pinned():
    # Changing the vocabulary, template order, or random draw order changes every prompt and
    # invalidates comparisons with earlier runs.
    first = build_prompt_set(1, SEED)[0]
    assert first["family"] == "field_note_extraction"
    assert first["prompt"] == (
        "ID0001. Find location: Morgan Lee will deliver the amber folder to Lakeside Office "
        "on October 19."
    )


def test_prompts_cycle_through_every_family_with_balanced_tiers():
    prompts = build_prompt_set(36, SEED)
    assert len({p["family"] for p in prompts}) == 18
    assert [p["family"] for p in prompts[:18]] == [p["family"] for p in prompts[18:]]
    tiers = Counter(p["intended_complexity"] for p in prompts)
    assert tiers == {"simple": 12, "moderate": 12, "complex": 12}


def test_every_generated_prompt_is_unique_and_fully_rendered():
    prompts = build_prompt_set(1_000, SEED)
    assert len({p["prompt"] for p in prompts}) == 1_000
    assert not any("{" in p["prompt"] or "}" in p["prompt"] for p in prompts)


def test_percentile_uses_the_nearest_rank_method():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 95) == 50.0
    assert percentile([], 50) is None


def make_summary() -> dict:
    prompts = build_prompt_set(3, SEED)
    rows = [
        {
            "prompt_hash": f"hash-{index}",
            "status_code": 200 if index < 2 else "error",
            "intended_complexity": prompt["intended_complexity"],
            "complexity": "simple" if index == 0 else "complex",
            "selected_model": "llama3.2:1b",
            "provider_latency_ms_simulated": 100.0 + index,
            "end_to_end_ms": 150.0 + index,
        }
        for index, prompt in enumerate(prompts)
    ]
    return summarize(
        prompts,
        rows,
        settings=RunSettings(
            "run-1",
            datetime(2026, 10, 20, 12, 0, tzinfo=UTC),
            datetime(2026, 10, 20, 12, 1, tzinfo=UTC),
            duration_s=2.0,
            concurrency=4,
            seed=SEED,
        ),
        stats={
            "input_tokens": 100,
            "output_tokens": 50,
            "actual_estimated_cost_usd": 0.0,
            "hypothetical_baseline_cost_usd": 0.00075,
            "estimated_savings_usd": 0.00075,
            "estimated_cost_reduction_pct": 100.0,
        },
        health={"status": "ok"},
        models=[{"model_id": "llama3.2:1b", "available_for_routing": True}],
        pricing={
            "models": {"gpt-4o": {"input": 2.5, "output": 10.0}},
            "last_verified": "2026-10-20",
        },
    )


def test_summary_separates_successes_from_failures():
    summary = make_summary()
    counts = (summary["request_count"], summary["success_count"], summary["failure_count"])
    assert counts == (3, 2, 1)
    assert summary["successful_requests_per_second"] == 1.0
    assert summary["routed_model_distribution"] == {"llama3.2:1b": 2}
    assert summary["intended_tier_matches"] == 1  # only the first row agrees with its label
    assert summary["intended_tier_agreement_pct"] == 50.0
    assert summary["token_usage_estimates"] == {"input": 100, "output": 50, "total": 150}
    assert summary["end_to_end_latency_ms"]["p50"] == 151.0


def test_report_states_its_evidence_class_and_is_driven_by_the_summary():
    report = render_report(make_summary())
    assert "simulated integration/load run; not live model inference" in report
    assert "should not be presented as a real savings result" in report
    assert "| Requests | 3 |" in report
    assert "| Successful / failed | 2 / 1 |" in report
    assert "| Hypothetical API-fee reduction | 100.00% |" in report
    assert "**Run ID:** `run-1`" in report
