from datetime import UTC, datetime

from app.core.sample_data import sample_requests


def test_sample_requests_are_clearly_synthetic():
    rows = sample_requests(datetime(2026, 10, 20, tzinfo=UTC))
    assert len(rows) == 42  # 3 requests a day for 14 days
    assert all(row["prompt_hash"] == "sample data" for row in rows)
    assert all(row["id"].startswith("sample-") for row in rows)
    assert rows[0]["created_at"].startswith("2026-10-07")
    assert rows[-1]["created_at"].startswith("2026-10-20")


def test_sample_requests_include_escalations_and_costs():
    rows = sample_requests(datetime(2026, 10, 20, tzinfo=UTC))
    escalated = [row for row in rows if row["escalated"]]
    assert len(escalated) == 3  # today, 5 days ago, and 10 days ago
    assert all(row["model_id"] == "claude-sonnet-5" for row in escalated)
    assert all(row["original_model_id"] == "gpt-4o-mini" for row in escalated)
    assert any(row["cost_usd"] > 0 for row in rows)
    local = [row for row in rows if row["original_model_id"] == "llama3.2:1b"]
    assert local and all(row["cost_usd"] == 0 for row in local)  # local + local verifier is free
