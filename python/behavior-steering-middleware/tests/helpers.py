"""Shared test utilities for behavior-steering middleware tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage


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


def make_mock_tool_request(
    tool_name: str,
    tool_call_id: str = "test_id",
    args: dict | None = None,
) -> MagicMock:
    """Create a mock ToolCallRequest for testing wrap_tool_call."""
    mock = MagicMock()
    mock.tool_call = {
        "name": tool_name,
        "id": tool_call_id,
        "args": args or {},
    }
    return mock


def make_mock_tool_handler(
    content: str = "Tool result",
    tool_call_id: str = "test_id",
) -> MagicMock:
    """Create a mock tool handler that returns a ToolMessage."""
    return MagicMock(
        return_value=ToolMessage(content=content, tool_call_id=tool_call_id)
    )
