"""Tests for the basic agent chat function."""

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from agents.basic_agent import chat
from core.errors import ResponseError


class TestChat:
    """Tests for chat."""

    def test_returns_assistant_content_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chat returns the assistant's message content on a successful API call."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Hello, world!"}}
            ]
        }
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        result = chat("Hi", client=mock_client)

        assert result == "Hello, world!"
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/chat/completions"
        payload: dict[str, Any] = call_args[1]["json"]
        assert payload["model"] == "test-model"
        assert payload["messages"][-1]["content"] == "Hi"

    def test_uses_explicit_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chat respects an explicitly provided model argument."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "default-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        chat("Hi", model="explicit-model", client=mock_client)
        payload: dict[str, Any] = mock_client.post.call_args[1]["json"]
        assert payload["model"] == "explicit-model"

    def test_raises_when_no_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chat raises ResponseError when the API returns no choices."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        with pytest.raises(ResponseError, match="no choices"):
            chat("Hi", client=mock_client)

    def test_raises_when_content_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chat raises ResponseError when the response lacks message content."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"choices": [{"message": {}}]}
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        with pytest.raises(ResponseError, match="missing message content"):
            chat("Hi", client=mock_client)

    def test_uses_custom_system_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chat includes the provided system prompt in the request."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        chat("Hi", system_prompt="You are a coder.", client=mock_client)
        payload: dict[str, Any] = mock_client.post.call_args[1]["json"]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are a coder."
