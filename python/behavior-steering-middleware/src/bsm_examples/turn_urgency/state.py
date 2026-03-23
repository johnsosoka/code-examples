"""Agent state schema for turn-based urgency middleware.

Defines the TypedDict state class that extends AgentState with urgency
tracking fields populated by TurnUrgencyMiddleware.
"""

from __future__ import annotations

from langchain.agents.middleware import AgentState
from typing_extensions import NotRequired


class UrgencyAgentState(AgentState):
    """Agent state schema with urgency tracking fields.

    Extends AgentState for use with create_agent and AgentMiddleware.
    All urgency fields are optional (NotRequired) since they are populated
    progressively as the middleware processes model calls.
    """

    urgency_level: NotRequired[str]
    turns_used: NotRequired[int]
    turns_remaining: NotRequired[int]
    is_terminal: NotRequired[bool]
    completion_tool_called: NotRequired[bool]
