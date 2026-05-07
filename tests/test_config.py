"""Tests for src.utils.config."""


import pytest

from utils.config import load_config


class TestLoadConfig:
    """Happy-path and override behaviour."""

    def test_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All required env vars set → Config returned with correct values."""
        monkeypatch.setenv("OPENCODE_API_KEY", "secret-key")
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_MODEL", "gpt-4o")

        cfg = load_config()

        assert cfg.api_key == "secret-key"
        assert cfg.base_url == "https://api.example.com"
        assert cfg.model == "gpt-4o"
        assert cfg.system_prompt == "You are a helpful assistant."
        assert cfg.timeout == 60.0

    def test_override_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keyword argument overrides the corresponding env var."""
        monkeypatch.setenv("OPENCODE_API_KEY", "secret-key")
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_MODEL", "gpt-4o")

        cfg = load_config(model="custom-model")

        assert cfg.model == "custom-model"
        assert cfg.api_key == "secret-key"
        assert cfg.base_url == "https://api.example.com"

    def test_defaults_when_not_overridden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """system_prompt and timeout fall back to defaults when omitted."""
        monkeypatch.setenv("OPENCODE_API_KEY", "k")
        monkeypatch.setenv("OPENCODE_BASE_URL", "http://localhost")
        monkeypatch.setenv("OPENCODE_MODEL", "m")

        cfg = load_config()

        assert cfg.system_prompt == "You are a helpful assistant."
        assert cfg.timeout == 60.0

    def test_override_system_prompt_and_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """system_prompt and timeout can be overridden via kwargs."""
        monkeypatch.setenv("OPENCODE_API_KEY", "k")
        monkeypatch.setenv("OPENCODE_BASE_URL", "http://localhost")
        monkeypatch.setenv("OPENCODE_MODEL", "m")

        cfg = load_config(system_prompt="Be concise.", timeout=30.0)

        assert cfg.system_prompt == "Be concise."
        assert cfg.timeout == 30.0

    def test_override_api_key_and_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Required fields can also be overridden via kwargs."""
        monkeypatch.setenv("OPENCODE_API_KEY", "env-key")
        monkeypatch.setenv("OPENCODE_BASE_URL", "http://env.local")
        monkeypatch.setenv("OPENCODE_MODEL", "env-model")

        cfg = load_config(api_key="override-key", base_url="http://override.local")

        assert cfg.api_key == "override-key"
        assert cfg.base_url == "http://override.local"
        assert cfg.model == "env-model"


class TestLoadConfigValidation:
    """Missing or empty required fields raise ValueError."""

    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unset all OPENCODE_* env vars that load_config reads."""
        for key in (
            "OPENCODE_API_KEY",
            "OPENCODE_BASE_URL",
            "OPENCODE_MODEL",
            "OPENCODE_SYSTEM_PROMPT",
            "OPENCODE_TIMEOUT",
        ):
            monkeypatch.delenv(key, raising=False)

    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing OPENCODE_API_KEY raises ValueError naming the key."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENCODE_BASE_URL", "http://localhost")
        monkeypatch.setenv("OPENCODE_MODEL", "m")

        with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
            load_config()

    def test_missing_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing OPENCODE_BASE_URL raises ValueError naming the key."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENCODE_API_KEY", "k")
        monkeypatch.setenv("OPENCODE_MODEL", "m")

        with pytest.raises(ValueError, match="OPENCODE_BASE_URL"):
            load_config()

    def test_missing_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing OPENCODE_MODEL raises ValueError naming the key."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENCODE_API_KEY", "k")
        monkeypatch.setenv("OPENCODE_BASE_URL", "http://localhost")

        with pytest.raises(ValueError, match="OPENCODE_MODEL"):
            load_config()

    def test_empty_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty string for OPENCODE_API_KEY is treated as missing."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENCODE_API_KEY", "")
        monkeypatch.setenv("OPENCODE_BASE_URL", "http://localhost")
        monkeypatch.setenv("OPENCODE_MODEL", "m")

        with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
            load_config()

    def test_empty_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty string for OPENCODE_BASE_URL is treated as missing."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENCODE_API_KEY", "k")
        monkeypatch.setenv("OPENCODE_BASE_URL", "")
        monkeypatch.setenv("OPENCODE_MODEL", "m")

        with pytest.raises(ValueError, match="OPENCODE_BASE_URL"):
            load_config()

    def test_empty_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty string for OPENCODE_MODEL is treated as missing."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENCODE_API_KEY", "k")
        monkeypatch.setenv("OPENCODE_BASE_URL", "http://localhost")
        monkeypatch.setenv("OPENCODE_MODEL", "")

        with pytest.raises(ValueError, match="OPENCODE_MODEL"):
            load_config()

    def test_whitespace_only_treated_as_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strings containing only whitespace are treated as empty."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENCODE_API_KEY", "   ")
        monkeypatch.setenv("OPENCODE_BASE_URL", "http://localhost")
        monkeypatch.setenv("OPENCODE_MODEL", "m")

        with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
            load_config()
