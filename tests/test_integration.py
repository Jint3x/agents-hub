"""Integration tests for the BasicAgent lifecycle."""

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agents import basic_agent
from core import AgentError, APIError, BasicAgent, ResponseError


class TestMultiTurnConversation:
    """History accumulates across multiple run() calls."""

    def test_three_turns_accumulate_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three user messages produce 7 history entries and full payload each turn."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = [
            {"choices": [{"message": {"content": "Reply 1"}}]},
            {"choices": [{"message": {"content": "Reply 2"}}]},
            {"choices": [{"message": {"content": "Reply 3"}}]},
        ]

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        agent = BasicAgent(client=mock_client)

        assert agent.run("Hello") == "Reply 1"
        assert agent.run("How are you?") == "Reply 2"
        assert agent.run("Goodbye") == "Reply 3"

        # 1 system + 3 user + 3 assistant = 7 entries
        assert len(agent.history) == 7
        assert agent.history[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert agent.history[1] == {"role": "user", "content": "Hello"}
        assert agent.history[2] == {"role": "assistant", "content": "Reply 1"}
        assert agent.history[3] == {"role": "user", "content": "How are you?"}
        assert agent.history[4] == {"role": "assistant", "content": "Reply 2"}
        assert agent.history[5] == {"role": "user", "content": "Goodbye"}
        assert agent.history[6] == {"role": "assistant", "content": "Reply 3"}

        # Verify each call sent the full accumulated history
        calls = mock_client.post.call_args_list
        assert len(calls) == 3

        first_payload: dict[str, Any] = calls[0][1]["json"]
        assert first_payload["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        second_payload: dict[str, Any] = calls[1][1]["json"]
        assert second_payload["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "How are you?"},
        ]

        third_payload: dict[str, Any] = calls[2][1]["json"]
        assert third_payload["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Reply 2"},
            {"role": "user", "content": "Goodbye"},
        ]


class TestHistoryReset:
    """reset_history() restores the initial system prompt."""

    def test_reset_restores_system_prompt_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After reset, history is just the system prompt; a new turn rebuilds from there."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = [
            {"choices": [{"message": {"content": "R1"}}]},
            {"choices": [{"message": {"content": "R2"}}]},
            {"choices": [{"message": {"content": "R3"}}]},
        ]

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        agent = BasicAgent(client=mock_client)

        agent.run("First")
        agent.run("Second")
        assert len(agent.history) == 5  # system + 2 user + 2 assistant

        agent.reset_history()
        assert len(agent.history) == 1
        assert agent.history[0] == {"role": "system", "content": "You are a helpful assistant."}

        agent.run("Third")
        assert len(agent.history) == 3  # system + 1 user + 1 assistant
        assert agent.history[1] == {"role": "user", "content": "Third"}
        assert agent.history[2] == {"role": "assistant", "content": "R3"}


class TestContextManagerLifecycle:
    """BasicAgent works as a context manager and cleans up owned clients."""

    def test_owned_client_closed_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An agent that creates its own client closes it on context exit."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        with patch("httpx.Client") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.post.return_value = mock_response
            mock_instance.is_closed = False
            mock_client_cls.return_value = mock_instance

            with BasicAgent() as agent:
                result = agent.run("Hello")
                assert result == "ok"

            assert mock_instance.close.called is True
            # Simulate the close effect for the assertion
            mock_instance.is_closed = True
            assert mock_instance.is_closed is True

    def test_borrowed_client_not_closed_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A borrowed client is left open after context exit."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        with BasicAgent(client=mock_client) as agent:
            agent.run("Hello")

        mock_client.close.assert_not_called()


class TestErrorPropagation:
    """HTTP and response errors are raised correctly and are catchable."""

    def test_http_429_raises_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An HTTP 429 raises APIError with status_code=429."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Too Many Requests",
            request=MagicMock(url="https://api.example.com/chat/completions"),
            response=mock_response,
        )

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        agent = BasicAgent(client=mock_client)

        with pytest.raises(APIError) as exc_info:
            agent.run("Hello")

        assert exc_info.value.status_code == 429
        assert isinstance(exc_info.value, AgentError)

    def test_empty_choices_raises_response_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A response with choices: [] raises ResponseError."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"choices": []}

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        agent = BasicAgent(client=mock_client)

        with pytest.raises(ResponseError) as exc_info:
            agent.run("Hello")

        assert exc_info.value.response_data == {"choices": []}
        assert isinstance(exc_info.value, AgentError)

    def test_both_errors_catchable_as_agent_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """APIError and ResponseError can both be caught as AgentError."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response_429 = MagicMock(spec=httpx.Response)
        mock_response_429.status_code = 429
        mock_response_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Too Many Requests",
            request=MagicMock(url="https://api.example.com/chat/completions"),
            response=mock_response_429,
        )

        mock_response_empty = MagicMock(spec=httpx.Response)
        mock_response_empty.status_code = 200
        mock_response_empty.raise_for_status.return_value = None
        mock_response_empty.json.return_value = {"choices": []}

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.side_effect = [
            mock_response_429,
            mock_response_empty,
        ]

        agent = BasicAgent(client=mock_client)

        caught: list[str] = []
        for _ in range(2):
            try:
                agent.run("Hello")
            except AgentError as e:
                caught.append(type(e).__name__)

        assert caught == ["APIError", "ResponseError"]


class TestChatBackwardCompatibility:
    """agents.basic_agent.chat() still works and delegates to BasicAgent."""

    def test_chat_returns_same_result_as_basic_agent_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """chat() returns the same string as creating a BasicAgent and calling run()."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello back"}}]
        }

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        direct = BasicAgent(client=mock_client)
        direct_result = direct.run("Hi")

        mock_client.reset_mock()
        mock_client.post.return_value = mock_response

        compat_result = basic_agent.chat("Hi", client=mock_client)

        assert compat_result == direct_result == "Hello back"

    def test_chat_accepts_same_keyword_arguments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """chat() accepts model, system_prompt, and client keywords."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        result = basic_agent.chat(
            "Hi",
            model="custom-model",
            system_prompt="You are a coder.",
            client=mock_client,
        )

        assert result == "ok"
        payload: dict[str, Any] = mock_client.post.call_args[1]["json"]
        assert payload["model"] == "custom-model"
        assert payload["messages"][0]["content"] == "You are a coder."


class TestCustomSystemPrompt:
    """A custom system prompt is sent as messages[0]."""

    def test_system_prompt_in_first_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Creating BasicAgent with system_prompt places it at messages[0]."""
        monkeypatch.setenv("OPENCODE_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
        monkeypatch.setenv("OPENCODE_MODEL", "test-model")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Arrr!"}}]
        }

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.post.return_value = mock_response

        agent = BasicAgent(
            system_prompt="You are a pirate.",
            client=mock_client,
        )
        agent.run("Say hello")

        payload: dict[str, Any] = mock_client.post.call_args[1]["json"]
        assert payload["messages"][0] == {
            "role": "system",
            "content": "You are a pirate.",
        }
