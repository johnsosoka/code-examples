"""Shared test utilities for behavior-steering middleware tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage


def make_mock_request(
    messages: list,
    system_message: SystemMessage | None = None,
) -> MagicMock:
    """Create a mock ModelRequest for testing wrap_model_call."""
    mock = MagicMock(spec=ModelRequest)
    mock.messages = messages
    mock.system_message = system_message
    mock.override.return_value = mock
    return mock


def make_mock_handler(response_messages: list | None = None) -> MagicMock:
    """Create a mock handler that returns a ModelResponse."""
    mock_response = MagicMock(spec=ModelResponse)
    mock_response.result = response_messages or [AIMessage(content="Response")]
    return MagicMock(return_value=mock_response)
