"""Tests for the env loader utility."""

import os
from pathlib import Path

import pytest

from agents_hub.utils.env import get_required_env, load_env


class TestLoadEnv:
    """Tests for load_env."""

    def test_loads_from_explicit_path(self, tmp_path: Path) -> None:
        """load_env reads variables from an explicit file path."""
        dotenv = tmp_path / ".env"
        dotenv.write_text("TEST_VAR=hello\n")
        load_env(dotenv)
        assert os.environ.get("TEST_VAR") == "hello"

    def test_searches_upward_when_no_path_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_env searches parent directories for .env when no path is given."""
        dotenv = tmp_path / ".env"
        dotenv.write_text("UPWARD_VAR=found\n")
        monkeypatch.chdir(tmp_path)
        load_env()
        assert os.environ.get("UPWARD_VAR") == "found"


class TestGetRequiredEnv:
    """Tests for get_required_env."""

    def test_returns_value_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_required_env returns the value of a set variable."""
        monkeypatch.setenv("REQUIRED_TEST", "value")
        assert get_required_env("REQUIRED_TEST") == "value"

    def test_raises_when_missing(self) -> None:
        """get_required_env raises RuntimeError when the variable is missing."""
        os.environ.pop("MISSING_VAR", None)
        with pytest.raises(
            RuntimeError, match="Missing required environment variable: MISSING_VAR"
        ):
            get_required_env("MISSING_VAR")

    def test_raises_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_required_env raises RuntimeError when the variable is empty."""
        monkeypatch.setenv("EMPTY_VAR", "")
        with pytest.raises(RuntimeError, match="Missing required environment variable: EMPTY_VAR"):
            get_required_env("EMPTY_VAR")
