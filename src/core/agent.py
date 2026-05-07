"""BasicAgent — reusable agent with conversation history."""

from __future__ import annotations

from typing import Any

import httpx

from core.errors import APIError, ResponseError
from utils.config import Config, load_config


class BasicAgent:
    """A minimal agent that tracks conversation history across turns.

    The agent can be used either with a borrowed :class:`httpx.Client` (the
    caller is responsible for closing it) or with an internally-created client
    that is closed automatically when the agent is used as a context manager.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        client: httpx.Client | None = None,
        config: Config | None = None,
    ) -> None:
        """Create a new BasicAgent.

        Args:
            model: Model identifier. Overrides the config/env value.
            system_prompt: Initial system message. Overrides the config/env value.
            client: Optional pre-configured HTTP client. When omitted, a client
                is created from environment variables and owned by this agent.
            config: Optional pre-built :class:`Config`. When omitted, config is
                loaded from environment variables.
        """
        self._config = config or load_config()
        self._model = model or self._config.model
        self._system_prompt = system_prompt or self._config.system_prompt

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                base_url=self._config.base_url,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._config.timeout,
            )
            self._owns_client = True

        self._history: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt}
        ]

    @property
    def history(self) -> list[dict[str, str]]:
        """The accumulated conversation history."""
        return self._history

    def reset_history(self) -> None:
        """Reset history to the initial system prompt."""
        self._history = [
            {"role": "system", "content": self._system_prompt}
        ]

    def run(self, message: str) -> str:
        """Send *message* to the API and return the assistant's reply.

        The message and reply are appended to :attr:`history`.

        Args:
            message: The user message.

        Returns:
            The assistant's response content.

        Raises:
            APIError: If the HTTP request fails.
            ResponseError: If the response has no choices or missing content.
        """
        self._history.append({"role": "user", "content": message})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": list(self._history),
        }

        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise APIError(
                f"API request failed: {exc.response.status_code}",
                status_code=exc.response.status_code,
                url=str(exc.request.url),
            ) from exc
        except httpx.HTTPError as exc:
            raise APIError(
                f"API request failed: {exc}",
                url=str(exc.request.url) if exc.request else None,
            ) from exc

        data = response.json()
        choices: list[dict[str, Any]] = data.get("choices", [])
        if not choices:
            raise ResponseError(
                "API response contained no choices",
                response_data=data,
            )

        msg: dict[str, Any] = choices[0].get("message", {})
        content = msg.get("content") or msg.get("reasoning_content")
        if not content:
            raise ResponseError(
                "API response missing message content",
                response_data=data,
            )

        assistant_content = str(content)
        self._history.append({"role": "assistant", "content": assistant_content})
        return assistant_content

    def __enter__(self) -> BasicAgent:
        """Enter the runtime context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the client if we own it."""
        if self._owns_client:
            self._client.close()
