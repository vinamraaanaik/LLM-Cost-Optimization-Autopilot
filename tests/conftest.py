"""Hermetic test environment and shared fixtures."""

import os
import tempfile
from pathlib import Path

import pytest

from app.core.database import Database

# app.main opens its database at import time. Point that at a temp file so running the tests
# never creates data/autopilot.sqlite3 in the working tree.
os.environ["LLM_AUTOPILOT_DB"] = str(Path(tempfile.mkdtemp()) / "import-time.sqlite3")

_CONFIG_ENV = (
    "LLM_ALLOW_PAID_PROVIDERS",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_VERIFICATION_MODE",
    "LLM_ROUTING_RULES_PATH",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Every test starts local-only, with no hosted credentials or custom routing file."""
    for name in _CONFIG_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def database(tmp_path) -> Database:
    return Database(tmp_path / "test.sqlite3")


@pytest.fixture
def api_db(monkeypatch, database) -> Database:
    """Point the API module at a throwaway database."""
    import app.main as main_module

    monkeypatch.setattr(main_module, "database", database)
    return database
