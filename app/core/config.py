"""Filesystem locations, shared limits, and environment-driven settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_MAX_OUTPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 16_384
MAX_PROMPT_CHARS = 50_000

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable; an unset variable falls back to ``default``."""
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in _TRUE_VALUES
