"""Basic agent that calls an OpenAI-compatible API and returns the response."""

import os
from collections.abc import Sequence
from typing import Any

import httpx
from dotenv import load_dotenv


def _default_client() -> httpx.Client:
    """Build an HTTP client configured from environment variables."""
    load_dotenv()
    base_url = os.environ["OPENCODE_BASE_URL"]
    api_key = os.environ["OPENCODE_API_KEY"]
    return httpx.Client(
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60.0,
    )


def chat(
    message: str,
    *,
    model: str | None = None,
    system_prompt: str = "You are a helpful assistant.",
    client: httpx.Client | None = None,
) -> str:
    """Send a single user message to the LLM API and return the assistant's reply.

    Args:
        message: The user message.
        model: Model identifier. Defaults to the ``OPENCODE_MODEL`` env var.
        system_prompt: System prompt sent with the request.
        client: Optional pre-configured HTTP client. A default client is
            created from environment variables when omitted.

    Returns:
        The content of the assistant's response message.

    Raises:
        RuntimeError: If the API returns an error or the response is malformed.
    """
    if client is None:
        client = _default_client()
        close_client = True
    else:
        close_client = False

    try:
        resolved_model = model or os.environ["OPENCODE_MODEL"]
        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
        }
        response = client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choices: Sequence[dict[str, Any]] = data.get("choices", [])
        if not choices:
            raise RuntimeError("API response contained no choices")

        content = (
            choices[0].get("message", {}).get("content")
            or choices[0].get("message", {}).get("reasoning_content")
        )
        if not content:
            raise RuntimeError("API response missing message content")

        return str(content)
    finally:
        if close_client:
            client.close()
