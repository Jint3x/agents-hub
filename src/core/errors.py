"""Custom exception hierarchy for agent-related errors."""

from typing import Any


class AgentError(Exception):
    """Base exception for all agent-related errors."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} — details: {self.details}"
        return self.message


class APIError(AgentError):
    """Raised when an HTTP call to the LLM API fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        url: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.status_code = status_code
        self.url = url

    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.url is not None:
            parts.append(f"url={self.url}")
        if self.details:
            parts.append(f"details={self.details}")
        return " — ".join(parts)


class ResponseError(AgentError):
    """Raised when the API returns a malformed or empty response."""

    def __init__(
        self,
        message: str,
        *,
        response_data: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.response_data = response_data

    def __str__(self) -> str:
        parts = [self.message]
        if self.response_data is not None:
            snippet = str(self.response_data)[:200]
            parts.append(f"response={snippet}")
        if self.details:
            parts.append(f"details={self.details}")
        return " — ".join(parts)
