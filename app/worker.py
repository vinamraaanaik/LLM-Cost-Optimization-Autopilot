"""Standalone worker that drains the SQLite verification queue.

Run with ``python -m app.worker``. Jobs are leased, so a crashed worker's job is picked up
again once its lease expires.
"""

import asyncio
import logging
import os

from app.core.catalog import MODEL_REGISTRY
from app.core.config import env_flag
from app.core.database import PENDING_VERIFICATION, Database
from app.core.quality import infer_request_type
from app.core.verification import verify_and_escalate
from app.providers.base import Response
from app.providers.unified import send_request

logger = logging.getLogger("llm_cost_autopilot.worker")


def _fail(database: Database, request: dict, reason: str) -> None:
    """Mark both the request and its job as failed so neither is retried forever."""
    database.update_verification(
        request["id"], status="error", request_type=request["request_type"], reason=reason
    )
    database.finish_verification_job(request["id"], error=reason)


async def process_job(database: Database, job: dict) -> None:
    request_id = job["request_id"]
    request = database.request_by_id(request_id)
    if request is None:
        database.finish_verification_job(request_id, error="Request row is missing.")
        return
    if request["verification_status"] not in PENDING_VERIFICATION:
        database.finish_verification_job(request_id)  # already settled elsewhere
        return

    original_model = MODEL_REGISTRY.get(request["original_model_id"])
    if original_model is None:
        missing = request["original_model_id"]
        _fail(database, request, f"Original model is no longer configured: {missing}.")
        return

    original_response = Response(
        output_text=request["initial_output_text"],
        input_tokens=int(request["input_tokens"]),
        output_tokens=int(request["output_tokens"]),
        latency_ms=float(request["latency_ms"]),
        cost_usd=float(request["cost_usd"]),
        model_id=original_model.model_id,
    )
    try:
        await verify_and_escalate(
            database=database,
            request_id=request_id,
            prompt=request["prompt"],
            request_type=infer_request_type(request["prompt"], request["request_type"]),
            complexity=request["complexity"],
            original_model=original_model,
            original_response=original_response,
            max_output_tokens=job["max_output_tokens"],
            send=send_request,
        )
    except Exception as error:
        logger.exception("Verification job failed for request %s", request_id)
        _fail(database, request, f"Verification worker failed ({type(error).__name__}).")
        return
    database.finish_verification_job(request_id)
    logger.info("Completed verification for request %s", request_id)


async def run_worker() -> None:
    if not env_flag("LLM_VERIFICATION_WORKER_ENABLED", default=True):
        logger.info("Verification worker is disabled by configuration.")
        return

    poll_seconds = max(0.1, float(os.getenv("LLM_VERIFICATION_WORKER_POLL_SECONDS", "0.5")))
    lease_seconds = max(30, int(os.getenv("LLM_VERIFICATION_JOB_LEASE_SECONDS", "600")))
    database = Database()
    logger.info("Verification worker started; polling every %.2f seconds.", poll_seconds)
    while True:
        job = database.claim_verification_job(lease_seconds=lease_seconds)
        if job is None:
            await asyncio.sleep(poll_seconds)
            continue
        await process_job(database, job)


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Verification worker stopped.")


if __name__ == "__main__":
    main()
