import pytest

from app.core.config import env_flag


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("YES", True), (" on ", True), ("0", False), ("", False)],
)
def test_env_flag_parses_truthy_and_falsy_values(monkeypatch, value, expected):
    monkeypatch.setenv("TEST_FLAG", value)
    assert env_flag("TEST_FLAG") is expected


def test_env_flag_uses_default_only_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_FLAG", raising=False)
    assert env_flag("TEST_FLAG") is False
    assert env_flag("TEST_FLAG", default=True) is True
