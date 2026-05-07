"""Unit tests for the custom exception hierarchy."""

import pytest

from core.errors import AgentError, APIError, ResponseError


class TestAgentError:
    def test_message_only(self) -> None:
        err = AgentError("something went wrong")
        assert err.message == "something went wrong"
        assert err.details is None
        assert str(err) == "something went wrong"

    def test_message_with_details(self) -> None:
        err = AgentError("something went wrong", details={"foo": "bar", "count": 42})
        assert err.message == "something went wrong"
        assert err.details == {"foo": "bar", "count": 42}
        assert str(err) == "something went wrong — details: {'foo': 'bar', 'count': 42}"

    def test_is_exception(self) -> None:
        with pytest.raises(AgentError):
            raise AgentError("boom")


class TestAPIError:
    def test_message_only(self) -> None:
        err = APIError("request failed")
        assert err.message == "request failed"
        assert err.status_code is None
        assert err.url is None
        assert err.details is None
        assert str(err) == "request failed"

    def test_with_status_code_and_url(self) -> None:
        err = APIError("request failed", status_code=500, url="https://api.example.com/v1/chat")
        assert err.status_code == 500
        assert err.url == "https://api.example.com/v1/chat"
        text = str(err)
        assert "request failed" in text
        assert "status=500" in text
        assert "url=https://api.example.com/v1/chat" in text

    def test_with_details(self) -> None:
        err = APIError("request failed", status_code=503, details={"retry_after": 10})
        text = str(err)
        assert "request failed" in text
        assert "status=503" in text
        assert "details={'retry_after': 10}" in text

    def test_is_agent_error(self) -> None:
        err = APIError("request failed")
        assert isinstance(err, AgentError)

    def test_caught_by_agent_error(self) -> None:
        with pytest.raises(AgentError):
            raise APIError("request failed")


class TestResponseError:
    def test_message_only(self) -> None:
        err = ResponseError("empty response")
        assert err.message == "empty response"
        assert err.response_data is None
        assert err.details is None
        assert str(err) == "empty response"

    def test_with_response_data(self) -> None:
        data = {"choices": [], "model": "gpt-4"}
        err = ResponseError("empty response", response_data=data)
        assert err.response_data == data
        text = str(err)
        assert "empty response" in text
        assert "response={'choices': [], 'model': 'gpt-4'}" in text

    def test_response_data_truncation(self) -> None:
        data = {"key": "x" * 500}
        err = ResponseError("too large", response_data=data)
        text = str(err)
        assert "too large" in text
        assert len(text) < 300

    def test_with_details(self) -> None:
        err = ResponseError("malformed", response_data={"a": 1}, details={"hint": "bad json"})
        text = str(err)
        assert "malformed" in text
        assert "response={'a': 1}" in text
        assert "details={'hint': 'bad json'}" in text

    def test_is_agent_error(self) -> None:
        err = ResponseError("malformed")
        assert isinstance(err, AgentError)

    def test_caught_by_agent_error(self) -> None:
        with pytest.raises(AgentError):
            raise ResponseError("malformed")


class TestInheritance:
    def test_api_error_is_agent_error(self) -> None:
        assert issubclass(APIError, AgentError)

    def test_response_error_is_agent_error(self) -> None:
        assert issubclass(ResponseError, AgentError)

    def test_catch_both_with_agent_error(self) -> None:
        caught = []
        for exc in [APIError("api fail"), ResponseError("response fail")]:
            try:
                raise exc
            except AgentError as e:
                caught.append(type(e).__name__)
        assert caught == ["APIError", "ResponseError"]
