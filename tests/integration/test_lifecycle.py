import asyncio

import app.main as main_module
from app.core import complexity
from tests.helpers import run


def run_lifespan_for(seconds: float) -> None:
    async def scenario():
        async with main_module.lifespan(main_module.app):
            assert complexity._serving_model is not None  # fitted before the first request
            await asyncio.sleep(seconds)

    run(scenario())


def test_startup_fits_the_classifier_with_recorded_failures(monkeypatch, api_db):
    seen = []
    monkeypatch.setattr(complexity, "_serving_model", None)
    monkeypatch.setattr(
        main_module,
        "load_serving_classifier",
        lambda failures: seen.append(failures) or complexity.load_serving_classifier(failures),
    )

    run_lifespan_for(0)

    assert seen == [[]]  # the (empty) routing-failure table was read at startup


def test_classifier_is_retrained_on_the_configured_schedule(monkeypatch, api_db):
    runs = []

    def fake_retrain(failures):
        runs.append(failures)
        return {"dataset_size": 270, "accuracy": 0.9}

    monkeypatch.setenv("LLM_RETRAIN_INTERVAL_SECONDS", "0.02")
    monkeypatch.setattr(main_module, "retrain", fake_retrain)

    run_lifespan_for(0.5)

    assert len(runs) >= 2


def test_a_failed_retrain_is_logged_and_the_schedule_continues(monkeypatch, api_db):
    attempts = []

    def failing_retrain(failures):
        attempts.append(failures)
        raise RuntimeError("boom")

    monkeypatch.setenv("LLM_RETRAIN_INTERVAL_SECONDS", "0.02")
    monkeypatch.setattr(main_module, "retrain", failing_retrain)

    run_lifespan_for(0.5)

    assert len(attempts) >= 2
