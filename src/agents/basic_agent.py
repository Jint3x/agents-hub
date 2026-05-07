"""Basic agent that calls an OpenAI-compatible API and returns the response."""

from __future__ import annotations

import httpx

from core.agent import BasicAgent


def chat(
    message: str,
    *,
    model: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
    client: httpx.Client | None = None,
) -> str:
    """Send a single user message to the LLM API and return the assistant's reply.

    This is a thin wrapper around :class:`core.BasicAgent` for backward
    compatibility.

    Args:
        message: The user message.
        model: Model identifier. Defaults to the ``OPENCODE_MODEL`` env var.
        system_prompt: System prompt sent with the request.
        client: Optional pre-configured HTTP client. A default client is
            created from environment variables when omitted.

    Returns:
        The content of the assistant's response message.
    """
    agent = BasicAgent(
        model=model,
        system_prompt=system_prompt,
        client=client,
    )
    return agent.run(message)
