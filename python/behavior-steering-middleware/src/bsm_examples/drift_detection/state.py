"""State type definitions for the Worker Check-In Monitor middleware."""

from __future__ import annotations

from langchain.agents.middleware import AgentState
from typing_extensions import NotRequired


class CheckInAgentState(AgentState):
    """Agent state extended with check-in tracking fields.

    These fields are written back into the graph state via
    ``ExtendedModelResponse`` after each model call so they can be
    inspected by other nodes or used for observability.
    """

    turns_since_check_in: NotRequired[int]
    intervention_count: NotRequired[int]
