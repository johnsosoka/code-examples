"""Agent state schema for the phase steering middleware."""

from __future__ import annotations

from langchain.agents.middleware import AgentState
from typing_extensions import NotRequired


class PhaseAgentState(AgentState):
    """Extends AgentState with phase tracking fields for observability.

    Fields are NotRequired so the state is valid before the middleware
    has run for the first time.

    Attributes:
        current_phase: Name of the currently active phase (e.g. "DISCOVERY").
        phase_iteration: Number of model calls completed in the current phase.
        total_iterations: Total model calls completed across all phases.
    """

    current_phase: NotRequired[str]
    phase_iteration: NotRequired[int]
    total_iterations: NotRequired[int]
