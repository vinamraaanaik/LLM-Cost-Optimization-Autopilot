import hashlib
import sqlite3

import pytest

from app.core.catalog import MODEL_REGISTRY
from app.core.database import Database
from tests.helpers import request_row, response


def test_persists_request_feedback_and_evaluations(database):
    database.record_request(request_row())
    assert database.add_feedback("r1", 5, "helpful")
    database.record_evaluation_run(
        "run1",
        [
            {
                "case_id": "case1",
                "model_id": "llama3.2:1b",
                "score": 1.0,
                "passed": True,
                "checks": [],
                "judge_score": 0.8,
                "judge_model_id": "llama3.2:1b",
                "judge_reason": "Mostly correct.",
                "model_cost_usd": 0.02,
                "judge_cost_usd": 0.01,
            },
        ],
    )

    recent = database.recent_requests()[0]
    assert recent["feedback_rating"] == 5
    assert recent["prompt_hash"] == hashlib.sha256(b"hello").hexdigest()
    assert database.cost_dashboard_requests()[0]["original_model_id"] == "llama3.2:1b"
    assert database.request_by_id("r1")["verification_status"] == "disabled"
    assert database.evaluation_scores() == [1.0]
    metrics = database.metrics()
    assert metrics["request_count"] == 1
    assert metrics["total_cost_usd"] == 0.03  # requests + golden-set model + judge spend
    assert metrics["evaluation_cost_usd"] == 0.02
    assert metrics["judge_cost_usd"] == 0.01
    assert metrics["average_feedback_rating"] == 5.0
    assert metrics["average_evaluation_score"] == 1.0
    assert metrics["average_judge_score"] == 0.8
    assert metrics["routing_failure_count"] == 0


def test_metrics_on_an_empty_database_are_zero_not_errors(database):
    metrics = database.metrics()
    assert metrics["request_count"] == 0
    assert metrics["total_cost_usd"] == 0.0
    assert metrics["average_feedback_rating"] == 0.0
    assert metrics["verification_pending_count"] == 0


def test_metrics_count_escalations_and_pending_verifications(database):
    for index, status in enumerate(["pending", "running", "verified", "disabled"]):
        database.record_request(
            request_row(id=f"r{index}", verification_status=status, cost_usd=0.001)
        )
    database.update_verification(
        "r2",
        status="escalated",
        request_type="general",
        reason="gap",
        verifier_response=response("llama3.1:8b", cost_usd=0.002),
        verifier_model=MODEL_REGISTRY["llama3.1:8b"],
        original_cost_usd=0.001,
        escalated=True,
    )
    metrics = database.metrics()
    assert metrics["verification_pending_count"] == 2
    assert metrics["escalation_count"] == 1
    assert metrics["escalation_cost_delta_usd"] == 0.001
    assert metrics["verification_cost_usd"] == 0.002


def test_feedback_rejects_unknown_request(database):
    assert not database.add_feedback("missing", 3)


def test_feedback_is_overwritten_not_duplicated(database):
    database.record_request(request_row())
    database.add_feedback("r1", 2, "meh")
    database.add_feedback("r1", 5, "better")
    assert database.metrics()["feedback_count"] == 1
    assert database.request_by_id("r1")["feedback_rating"] == 5


def test_update_verification_rejects_unknown_requests(database):
    with pytest.raises(ValueError, match="Unknown request_id"):
        database.update_verification("nope", status="verified", request_type="general", reason="x")


def test_database_uses_wal_mode_for_concurrent_processes(tmp_path):
    path = tmp_path / "wal.sqlite3"
    Database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_older_databases_are_upgraded_in_place(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE requests (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, prompt TEXT NOT NULL,
            model_id TEXT NOT NULL, provider_name TEXT NOT NULL, complexity TEXT NOT NULL,
            confidence REAL NOT NULL, route_reason TEXT NOT NULL, used_fallback INTEGER NOT NULL,
            output_text TEXT NOT NULL, input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL, latency_ms REAL NOT NULL, cost_usd REAL NOT NULL
        )"""
    )
    connection.execute(
        """INSERT INTO requests VALUES
        ('legacy', '2026-01-01T00:00:00+00:00', 'old prompt', 'llama3.2:1b', 'ollama',
         'simple', 0.9, 'test', 0, 'ok', 2, 1, 20, 0)"""
    )
    connection.commit()
    connection.close()

    legacy = Database(path).request_by_id("legacy")

    assert legacy["prompt_hash"] == hashlib.sha256(b"old prompt").hexdigest()
    assert legacy["original_model_id"] == "llama3.2:1b"
    assert legacy["initial_output_text"] == "ok"
    assert legacy["verification_status"] == "disabled"


def test_verification_jobs_are_leased_and_completed(database):
    database.record_request(request_row())
    database.enqueue_verification_job("r1", max_output_tokens=96)

    job = database.claim_verification_job(lease_seconds=60)

    assert job == {"request_id": "r1", "max_output_tokens": 96, "attempts": 1}
    assert database.claim_verification_job() is None  # leased, so nobody else can take it
    database.finish_verification_job("r1")
    assert database.verification_job("r1")["status"] == "completed"


def test_expired_leases_are_reclaimed(database):
    database.record_request(request_row())
    database.enqueue_verification_job("r1", max_output_tokens=96)
    assert database.claim_verification_job(lease_seconds=-1)["attempts"] == 1
    assert database.claim_verification_job()["attempts"] == 2  # the dead worker's job is retried


def test_enqueue_is_idempotent_and_failures_are_recorded(database):
    database.record_request(request_row())
    database.enqueue_verification_job("r1", 96)
    database.enqueue_verification_job("r1", 999)
    assert database.verification_job("r1")["max_output_tokens"] == 96
    database.finish_verification_job("r1", error="boom")
    job = database.verification_job("r1")
    assert (job["status"], job["last_error"]) == ("failed", "boom")
    with pytest.raises(ValueError, match="Unknown verification job"):
        database.finish_verification_job("missing")


def test_cost_usage_summary_covers_requests_only(database):
    database.record_request(request_row())
    assert database.cost_usage_summary() == {
        "request_count": 1,
        "input_tokens": 2,
        "output_tokens": 3,
        "actual_cost_usd": 0.0,
    }
