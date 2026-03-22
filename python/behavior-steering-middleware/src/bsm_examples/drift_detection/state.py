"""State type definitions for drift detection middleware."""

from __future__ import annotations

from langchain.agents.middleware import AgentState
from typing_extensions import NotRequired


class DriftAgentState(AgentState):
    drift_score: NotRequired[float]
    intervention_count: NotRequired[int]
    drift_report: NotRequired[dict[str, object]]
