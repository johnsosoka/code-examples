"""State type definitions for drift detection middleware."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class DriftSteeringState(TypedDict):
    """
    State for an agent with drift detection steering.

    Extends basic message state with drift detection metadata
    for monitoring and debugging steering behavior.
    """

    # Core message history (LangGraph standard)
    messages: Annotated[list[AnyMessage], add_messages]

    # Drift detection metadata
    drift_score: float  # Current drift score (0.0-1.0)
    intervention_count: int  # Number of interventions so far
    last_intervention_turn: int | None  # Turn number of last intervention

    # Tool usage tracking
    tool_call_history: list[dict]  # Record of tool calls per turn

    # Debug/diagnostics
    steering_events: list[dict]  # Log of steering decisions
