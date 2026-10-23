from app import worker
from tests.helpers import request_row, run


def queue_request(database, **overrides):
    fields = {
        "id": "worker-request",
        "prompt": "Reply with 42.",
        "output_text": "42",
        "original_model_id": "llama3.2:1b",
        "verification_status": "pending",
        **overrides,
    }
    database.record_request(request_row(**fields))
    database.enqueue_verification_job("worker-request", max_output_tokens=32)
    return database.claim_verification_job()


def test_worker_processes_a_queued_request(monkeypatch, database):
    job = queue_request(database)
    seen = {}

    async def fake_verify(**kwargs):
        seen.update(kwargs)
        database.update_verification(
            kwargs["request_id"],
            status="verified",
            request_type="general",
            quality_score=1.0,
            reason="Mock deterministic verification.",
        )

    monkeypatch.setattr(worker, "verify_and_escalate", fake_verify)
    run(worker.process_job(database, job))

    assert seen["request_id"] == "worker-request"
    assert seen["max_output_tokens"] == 32
    assert seen["original_response"].output_text == "42"
    assert database.request_by_id("worker-request")["verification_status"] == "verified"
    assert database.verification_job("worker-request")["status"] == "completed"


def test_worker_marks_request_and_job_failed_when_verification_crashes(monkeypatch, database):
    job = queue_request(database)

    async def crashing_verify(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(worker, "verify_and_escalate", crashing_verify)
    run(worker.process_job(database, job))

    assert database.request_by_id("worker-request")["verification_status"] == "error"
    queued = database.verification_job("worker-request")
    assert queued["status"] == "failed" and "RuntimeError" in queued["last_error"]


def test_worker_fails_jobs_whose_original_model_was_removed(database):
    job = queue_request(database, original_model_id="retired-model")

    run(worker.process_job(database, job))

    assert "no longer configured" in database.verification_job("worker-request")["last_error"]
    assert database.request_by_id("worker-request")["verification_status"] == "error"


def test_worker_skips_requests_that_were_already_settled(database):
    job = queue_request(database)
    database.update_verification(
        "worker-request", status="verified", request_type="general", reason=""
    )

    run(worker.process_job(database, job))

    assert database.verification_job("worker-request")["status"] == "completed"


def test_worker_fails_jobs_with_no_request_row(database):
    # Foreign keys normally cascade-delete the job too; force the orphan this branch guards.
    job = {"request_id": "ghost", "max_output_tokens": 32, "attempts": 1}
    database.record_request(request_row(id="ghost"))
    database.enqueue_verification_job("ghost", 32)
    with database._connection() as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DELETE FROM requests WHERE id = 'ghost'")

    run(worker.process_job(database, job))

    assert database.verification_job("ghost")["last_error"] == "Request row is missing."
