"""Tool-Triggered Phase Steering middleware for LangGraph agents.

Guides agent behavior through DISCOVERY -> CURATION -> RECOMMENDATION -> COMPLETE
phases where transitions are triggered by specific tool calls.
"""

from __future__ import annotations

from bsm_examples.phase_steering.detector import ToolTriggeredDetector
from bsm_examples.phase_steering.middleware import PhaseSteeringMiddleware
from bsm_examples.phase_steering.models import (
    Phase,
    PhaseBehavior,
    PhaseConfig,
    PhaseTransitionRule,
)
from bsm_examples.phase_steering.state import PhaseAgentState

__all__ = [
    "Phase",
    "PhaseBehavior",
    "PhaseConfig",
    "PhaseTransitionRule",
    "PhaseAgentState",
    "ToolTriggeredDetector",
    "PhaseSteeringMiddleware",
]
