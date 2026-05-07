"""Centralised configuration loader for agents-hub."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Runtime configuration for API-backed agents."""

    api_key: str
    base_url: str
    model: str
    system_prompt: str = "You are a helpful assistant."
    timeout: float = 60.0


def _require_env(name: str, value: str | None) -> str:
    """Return *value* if it is present and non-empty, otherwise raise ValueError."""
    if value is None or value.strip() == "":
        raise ValueError(f"Missing or empty required configuration: {name}")
    return value


def load_config(**overrides: str | float) -> Config:
    """Build a :class:`Config` from environment variables with optional overrides.

    Reads ``OPENCODE_API_KEY``, ``OPENCODE_BASE_URL``, and ``OPENCODE_MODEL``
    from the process environment.  Keyword arguments override the corresponding
    environment variable values.

    Args:
        **overrides: Keyword values to override environment variables.
            Recognised keys: ``api_key``, ``base_url``, ``model``,
            ``system_prompt``, ``timeout``.

    Returns:
        A fully populated :class:`Config` instance.

    Raises:
        ValueError: If ``api_key``, ``base_url``, or ``model`` is missing or
            empty after applying overrides.
    """
    api_key_override: str | None = overrides.get("api_key")  # type: ignore[assignment]
    base_url_override: str | None = overrides.get("base_url")  # type: ignore[assignment]
    model_override: str | None = overrides.get("model")  # type: ignore[assignment]
    system_prompt_override: str | None = overrides.get("system_prompt")  # type: ignore[assignment]
    timeout_override: float | None = overrides.get("timeout")  # type: ignore[assignment]

    api_key = _require_env(
        "OPENCODE_API_KEY",
        api_key_override or os.environ.get("OPENCODE_API_KEY"),
    )
    base_url = _require_env(
        "OPENCODE_BASE_URL",
        base_url_override or os.environ.get("OPENCODE_BASE_URL"),
    )
    model = _require_env(
        "OPENCODE_MODEL",
        model_override or os.environ.get("OPENCODE_MODEL"),
    )

    system_prompt = (
        system_prompt_override
        or os.environ.get("OPENCODE_SYSTEM_PROMPT")
        or "You are a helpful assistant."
    )

    timeout_raw = timeout_override or os.environ.get("OPENCODE_TIMEOUT")
    timeout: float = 60.0
    if timeout_raw is not None:
        timeout = float(timeout_raw)

    return Config(
        api_key=str(api_key),
        base_url=str(base_url),
        model=str(model),
        system_prompt=str(system_prompt),
        timeout=timeout,
    )
