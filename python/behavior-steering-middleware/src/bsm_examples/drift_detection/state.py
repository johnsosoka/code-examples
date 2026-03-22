"""State type definitions for the Worker Heartbeat Monitor middleware."""

from __future__ import annotations

from langchain.agents.middleware import AgentState
from typing_extensions import NotRequired


class HeartbeatAgentState(AgentState):
    """Agent state extended with heartbeat tracking fields.

    These fields are written back into the graph state via
    ``ExtendedModelResponse`` after each model call so they can be
    inspected by other nodes or used for observability.
    """

    turns_since_heartbeat: NotRequired[int]
    intervention_count: NotRequired[int]
